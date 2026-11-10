"""Interaction policies over a Belief.

Actions
  ask(slot)  ask the user which reading of a slot they mean (answer may be "no preference")
  show       show the user the result of the current most likely reading; the user accepts or
             rejects it (this is the only channel for slots the user cannot answer about, e.g.
             element type or a dropped condition, and is the analogue of VisInteract's visual check)
  submit     return the most likely result class

Objective (doc, part 3):  E[1(submitted class == true class)] - lam * #asks - mu * #shows,
with at most `budget` interactions (asks + shows). Users answer "no preference" with probability
`noise`; show judgements are taken as reliable.

  NoAsk     submit immediately
  AskAll    ask every askable slot that still has more than one reading, fixed order
  Natural   fixed order, stop as soon as the result class is certain
  Greedy    largest EC2 gain first; stop when certain, out of budget, or when no single
            interaction is worth its cost in immediate accuracy (myopic by design)
  ShowLoop  keep showing the most likely result until accepted (VisInteract-style feedback only)
  Optimal   exact dynamic programme over alive sets: the upper bound for any planner
  MCTS      POMCP-style search over belief states with transposition merging (states are alive
            sets, so different interaction orders that reach the same knowledge share statistics);
            PUCT prior from EC2 gain, greedy rollouts; runs entirely on cached results
All policies that ask may also show when `show_cost` is set and nothing askable is left.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

from .belief import Belief


@dataclass(frozen=True)
class Action:
    kind: str                # "ask" | "show" | "submit"
    slot: str | None = None

    def __str__(self):
        return self.kind if self.slot is None else "%s(%s)" % (self.kind, self.slot)


SUBMIT = Action("submit")
SHOW = Action("show")


@dataclass
class Setting:
    belief: Belief
    askable: list[str]                 # slots the policy may ask about
    budget: int | None = None          # max interactions (asks + shows); None = generous cap
    lam: float = 0.0                   # cost per ask
    noise: float = 0.0                 # P(user answers "no preference")
    show_cost: float | None = None     # cost per show; None = show disabled

    def left(self, used: int) -> int:
        if self.budget is not None:
            cap = self.budget
        else:
            cap = len(self.askable) * (1 if self.noise == 0 else 3) + (3 if self.show_cost is not None else 0)
        return max(0, cap - used)

    asks_left = left                   # older name

    def candidates(self, alive) -> list[str]:
        return [s for s in self.askable if self.belief.splits(alive, s)]

    def can_show(self, alive) -> bool:
        return self.show_cost is not None and not self.belief.certain(alive)


def one_step_ask(st: Setting, alive, slot: str) -> float:
    """Immediate accuracy gain of asking `slot`, net of its cost."""
    b = st.belief
    tot, now = b.mass(alive), b.acc_stop(alive)
    after = sum(b.mass(h) / tot * b.acc_stop(h) for h in b.split(alive, slot).values())
    return (1 - st.noise) * (after - now) - st.lam


def one_step_show(st: Setting, alive) -> float:
    b = st.belief
    p = b.acc_stop(alive)
    rest = b.after_show(alive, b.map_class(alive), False)
    return p + (1 - p) * (b.acc_stop(rest) if rest else 0.0) - p - st.show_cost


class Policy:
    name = "policy"

    def act(self, st: Setting, alive, used: int) -> Action:
        raise NotImplementedError

    @staticmethod
    def fallback(st, alive):
        """Nothing worth asking: show if allowed and still uncertain, else submit."""
        return SHOW if st.can_show(alive) and one_step_show(st, alive) > 0 else SUBMIT


class NoAsk(Policy):
    name = "no_ask"

    def act(self, st, alive, used):
        return SUBMIT


class AskAll(Policy):
    """Asks every askable slot that still has more than one reading left, in the fixed slot order, whether or not it
    changes the result; a slot answered with "no preference" stays unresolved and is asked again (same re-ask rule as
    every other policy). Submits when no slot is left or the budget is used up."""
    name = "ask_all"

    def act(self, st, alive, used):
        if st.left(used) == 0:
            return SUBMIT
        cands = st.candidates(alive)
        return Action("ask", cands[0]) if cands else SUBMIT


class Natural(Policy):
    name = "natural"

    def act(self, st, alive, used):
        if st.belief.certain(alive) or st.left(used) == 0:
            return SUBMIT
        cands = st.candidates(alive)
        return Action("ask", cands[0]) if cands else self.fallback(st, alive)


class Greedy(Policy):
    name = "greedy_ec2"

    def act(self, st, alive, used):
        b = st.belief
        if b.certain(alive) or st.left(used) == 0:
            return SUBMIT
        cands = st.candidates(alive)
        show = one_step_show(st, alive) if st.can_show(alive) else -math.inf
        ask = max((one_step_ask(st, alive, v) for v in cands), default=-math.inf)
        if not cands or (show > 0 and show > ask):
            return SHOW if show > 0 else SUBMIT
        if st.lam > 0 and ask < 0:
            return SUBMIT
        return Action("ask", max(cands, key=lambda v: (b.gain(alive, v), -st.askable.index(v))))


def coalition_gain(b, alive, U):
    """Delta_U(S): accuracy gain of submitting after learning every slot in U (noise-free)."""
    if not U:
        return 0.0
    groups = {}
    for i in alive:
        groups.setdefault(tuple(b.value(i, v) for v in U), []).append(i)
    tot = b.mass(alive)
    return sum(b.mass(g) / tot * b.acc_stop(tuple(g)) for g in groups.values()) - b.acc_stop(alive)


def shapley_values(b, alive, slots, m=None):
    """Exact Shapley values of the slots in the cooperative game v(U) = Delta_U(S) (all 2^|slots| coalitions).
    Efficiency: sum_v phi_v = Delta_slots(S); null player: a slot that never changes the result class gets 0.
    m < |slots| (budget truncation, the protocol): the semivalue that averages a slot's marginal
    contribution over coalitions of at most m - 1 other slots (uniform over sizes, then over coalitions of a size),
    i.e. only over coalitions that m remaining questions can complete; m >= |slots| gives the Shapley value."""
    from itertools import combinations
    from math import comb, factorial
    if m is not None and m < len(slots):
        key = ("shapm", alive, tuple(slots), m)
        if key in b._cache:
            return b._cache[key]
        n = len(slots)
        phi = {}
        for v in slots:
            others = [u for u in slots if u != v]
            t = 0.0
            for k in range(m):
                for U in combinations(others, k):
                    t += (coalition_gain(b, alive, U + (v,)) - coalition_gain(b, alive, U)) / (m * comb(n - 1, k))
            phi[v] = t
        b._cache[key] = phi
        return phi
    key = ("shap", alive, tuple(slots))
    if key in b._cache:
        return b._cache[key]
    n = len(slots)
    val = {}
    for k in range(n + 1):
        for U in combinations(slots, k):
            val[frozenset(U)] = coalition_gain(b, alive, U)
    w = [factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)]
    phi = {}
    for v in slots:
        others = [u for u in slots if u != v]
        t = 0.0
        for k in range(n):
            for U in combinations(others, k):
                fu = frozenset(U)
                t += w[k] * (val[fu | {v}] - val[fu])
        phi[v] = t
    b._cache[key] = phi
    return phi


class ShapleyGreedy(Policy):
    """Ask the slot with the largest Shapley value; submit when no slot's Shapley value pays for a question,
    i.e. (1 - eps) * phi_v - lam <= 0 for every askable slot (cf. greedy's one-step rule (1 - eps) * Delta_v - lam <= 0).
    truncate=True: Shapley values truncated to coalitions the remaining budget can complete (shapley_values(m=k))."""
    name = "shapley_greedy"

    def __init__(self, truncate: bool = False):
        self.truncate = truncate

    def act(self, st, alive, used):
        b = st.belief
        if b.certain(alive) or st.left(used) == 0:
            return SUBMIT
        cands = st.candidates(alive)
        if not cands:
            return SUBMIT
        phi = shapley_values(b, alive, cands, st.left(used) if self.truncate and st.budget is not None else None)
        top = max(phi[s] for s in cands)          # EC2 gain only breaks exact ties in phi (same choice as ranking by the triple)
        tied = [s for s in cands if phi[s] == top]
        v = tied[0] if len(tied) == 1 else max(tied, key=lambda s: (b.gain(alive, s), -st.askable.index(s)))
        if (1 - st.noise) * phi[v] - st.lam <= 0:
            return SUBMIT
        return Action("ask", v)



class ShapleyRollout(Policy):
    """One-step rollout improvement of ShapleyGreedy (Bertsekas): at each state take the action whose value, when
    followed by ShapleyGreedy, is largest (computed exactly on the belief). By the policy-improvement theorem its value
    under the belief is >= ShapleyGreedy's."""
    name = "shapley_rollout"

    def __init__(self, base: str = "shapley", truncate: bool = False):
        self.base = ShapleyGreedy(truncate) if base == "shapley" else Greedy()   # "greedy": ablation, EC2 one-step base policy
        self.truncate = truncate
        self._memo, self._key = {}, None

    def _v(self, st, alive, k):
        key = (alive, k)
        if key in self._memo:
            return self._memo[key]
        b = st.belief
        a = self.base.act(st, alive, st.left(0) - k) if k > 0 else SUBMIT
        v = b.acc_stop(alive) if a.kind != "ask" else self._q(st, alive, k, a.slot)
        self._memo[key] = v
        return v

    def _q(self, st, alive, k, slot):
        b = st.belief
        tot = b.mass(alive)
        ans = sum(b.mass(h) / tot * self._v(st, h, k - 1) for h in b.split(alive, slot).values())
        return -st.lam + st.noise * self._v(st, alive, k - 1) + (1 - st.noise) * ans

    def act(self, st, alive, used):
        b = st.belief
        key = (id(b), st.budget, st.lam, st.noise)
        if key != self._key:
            self._memo, self._key = {}, key
        k = st.left(used)
        if b.certain(alive) or k == 0:
            return SUBMIT
        cands = st.candidates(alive)
        if not cands:
            return SUBMIT
        phi = shapley_values(b, alive, cands, k if self.truncate and st.budget is not None else None)
        best, val = SUBMIT, b.acc_stop(alive)
        for v in sorted(cands, key=lambda s: -phi[s]):
            q = self._q(st, alive, k, v)
            if q > val + 1e-12:
                best, val = Action("ask", v), q
        return best

class EIGReadings(Policy):
    """Baseline after Qiu et al. 2025 (interactive Text-to-SQL via expected information gain): ask
    the slot with the largest expected entropy drop over candidate READINGS (not result classes),
    until the most likely reading reaches confidence `tau` or nothing is left to ask. It does not
    know which readings return the same result, so it can spend questions on differences that do
    not matter."""
    name = "eig_readings"

    def __init__(self, tau: float = 0.9):
        self.tau = tau

    @staticmethod
    def _H(b, alive):
        tot = b.mass(alive)
        return -sum(b.w[i] / tot * math.log(b.w[i] / tot) for i in alive if b.w[i] > 0)

    def act(self, st, alive, used):
        b = st.belief
        if st.left(used) == 0:
            return SUBMIT
        tot = b.mass(alive)
        if max(b.w[i] for i in alive) / tot >= self.tau:
            return SUBMIT
        cands = st.candidates(alive) or [s for s in st.askable if len(b.split(alive, s)) > 1]
        if not cands:
            return self.fallback(st, alive)
        h0 = self._H(b, alive)
        def ig(v):
            return h0 - sum(b.mass(h) / tot * self._H(b, h) for h in b.split(alive, v).values())
        return Action("ask", max(cands, key=lambda v: (ig(v), -st.askable.index(v))))


class EIGClasses(EIGReadings):
    """Mechanism control for EIGReadings: the same one-step expected entropy drop and confidence stop, but the
    entropy is taken over RESULT CLASSES (it knows which readings return the same objects)."""
    name = "eig_classes"

    def __init__(self, tau: float = 0.9, stop: str = "tau"):
        super().__init__(tau)
        self.stop = stop             # "tau": submit once the top class reaches tau; "delta": ask the max-IG slot only
                                     # if its one-step net gain (1 - eps) * Delta_v - lam is positive

    @staticmethod
    def _Hc(b, alive):
        tot = b.mass(alive)
        m = {}
        for i in alive:
            m[b.cls[i]] = m.get(b.cls[i], 0.0) + b.w[i] / tot
        return -sum(x * math.log(x) for x in m.values() if x > 0)

    def act(self, st, alive, used):
        b = st.belief
        if st.left(used) == 0 or b.certain(alive):
            return SUBMIT
        if self.stop == "tau" and b.acc_stop(alive) >= self.tau:
            return SUBMIT
        cands = st.candidates(alive)
        if not cands:
            return SUBMIT
        tot = b.mass(alive)
        h0 = self._Hc(b, alive)
        def ig(v):
            return h0 - sum(b.mass(h) / tot * self._Hc(b, h) for h in b.split(alive, v).values())
        v = max(cands, key=lambda v: (ig(v), -st.askable.index(v)))
        if self.stop == "delta" and (1 - st.noise) * coalition_gain(b, alive, (v,)) - st.lam <= 0:
            return SUBMIT
        return Action("ask", v)


class ShowLoop(Policy):
    name = "show_loop"

    def act(self, st, alive, used):
        return SHOW if st.left(used) > 0 and st.can_show(alive) else SUBMIT


class Optimal(Policy):
    """Exact expected-value maximisation. Exponential in general; fine for a few slots."""
    name = "optimal"

    def __init__(self, canonical: bool = False):
        self._key = None
        self.canonical = canonical      # memoise by canonical form (Belief.canon): exact, far fewer states

    def _solver(self, st: Setting):
        b, eps, lam, mu = st.belief, st.noise, st.lam, st.show_cost

        def q_ask(alive, k, s, V):
            tot = b.mass(alive)
            ans = sum(b.mass(h) / tot * V(h, k - 1) for h in b.split(alive, s).values())
            return -lam + eps * V(alive, k - 1) + (1 - eps) * ans

        def q_show(alive, k, V):
            p = b.acc_stop(alive)
            rest = b.after_show(alive, b.map_class(alive), False)
            return -mu + p + (1 - p) * (V(rest, k - 1) if rest else 0.0)

        memo = {}

        def V(alive, k):
            mk = (b.canon(alive) if self.canonical else alive, k)
            if mk in memo:
                return memo[mk]
            memo[mk] = v = _V(alive, k)
            return v

        def _V(alive, k):
            best = b.acc_stop(alive)
            if k == 0 or b.certain(alive):
                return best
            for s in st.candidates(alive):
                best = max(best, q_ask(alive, k, s, V))
            if st.can_show(alive):
                best = max(best, q_show(alive, k, V))
            return best

        def Q(alive, k, a):
            if a.kind == "submit":
                return b.acc_stop(alive)
            if a.kind == "show":
                return q_show(alive, k, V)
            return q_ask(alive, k, a.slot, V)

        return V, Q

    def _ensure(self, st):
        key = (id(st.belief), tuple(st.askable), st.budget, st.lam, st.noise, st.show_cost)
        if key != self._key:
            self._key, (self._V, self._Q) = key, self._solver(st)

    def value(self, st: Setting, alive=None, used: int = 0) -> float:
        self._ensure(st)
        return self._V(alive if alive is not None else st.belief.root, st.left(used))

    def act(self, st, alive, used):
        self._ensure(st)
        k = st.left(used)
        if k == 0 or st.belief.certain(alive):
            return SUBMIT
        acts = [SUBMIT] + [Action("ask", s) for s in st.candidates(alive)] + ([SHOW] if st.can_show(alive) else [])
        return max(acts, key=lambda a: (round(self._Q(alive, k, a), 12), a.kind == "submit"))


@dataclass
class _Node:
    n: int = 0
    v: float = 0.0                                # sum of returns through this node
    stats: dict = field(default_factory=dict)     # action -> [visits, value_sum]
    prior: dict = field(default_factory=dict)


class MCTS(Policy):
    """POMCP-style search over belief states. Submissions are scored by their success probability
    under the current belief (expected=True), which removes most simulation noise.
    Switches (all on = TopoMCTS):
      prior      "gain": PUCT prior from graph-node importance (EC2 gain); "uniform": none
      frontier   only ask a slot once the slots its candidates depend on are settled (graph edges)
      rave       share per-graph-node action values across the tree (all-moves-as-first)
      transpose  key states by what is known (alive set): different question orders merge into a
                 DAG; False keys by interaction history, i.e. a plain tree as in VisInteract
    """
    name = "mcts"

    def __init__(self, n_sim: int = 400, c_puct: float = 1.0, seed: int = 0, prior: str = "gain",
                 frontier: bool = False, rave: bool = False, transpose: bool = True, rave_k: float = 30.0,
                 expected: bool = True, rollout: str = "greedy", classes: bool = True,
                 abstract: bool = False, name: str | None = None, la_depth: int | None = None, plan_temp: float = 1.0):
        self.n_sim = n_sim
        self.c = c_puct
        self.rng = random.Random(seed)
        self.prior_mode = prior
        self.frontier = frontier
        self.rave = rave
        self.transpose = transpose
        self.la_depth = la_depth     # prior "lookahead": follow the greedy continuation for at most this many
                                     # steps, then score the state by acc(S) (None = to the end)
        self.abstract = abstract     # key states by their canonical form (Belief.canon): states that are
                                     # the same problem up to renaming share one node
        self.rave_k = rave_k
        self.plan_temp = plan_temp   # < 1: plan on a tempered belief p^beta (submission keeps the calibrated prior)
        self.classes = classes       # False: plan over raw readings, not knowing which readings return
                                     # the same result (the "no topology" ablation/baseline)
        self._shadow = {}
        self.rollout = rollout       # "greedy": continue simulated dialogues with the topology's
                                     # importance-ordered greedy policy; "random": uniform actions;
                                     # "order": ask in importance order until the class is certain (no
                                     # one-step stopping rule: stopping is left to the tree)
        self.expected = expected     # score a submission by its success probability under the
                                     # belief instead of a sampled 0/1 (variance reduction)
        if name:
            self.name = name
        self.greedy = Greedy()
        if prior == "lookahead_shapley":      # look-ahead prior whose continuation is Shapley greedy, not EC2 greedy
            self.greedy = ShapleyGreedy()
            self.prior_mode = "lookahead"

    # -- graph-aware pieces -------------------------------------------------------------------
    def _legal_asks(self, st, alive):
        cands = st.candidates(alive)
        if not self.frontier:
            return cands
        b, topo = st.belief, st.belief.topo
        out = []
        for v in cands:
            deps = [p for p, e in topo.slots[v].parents if e in ("candidates", "exists") and p in b.slots]
            if all(len(b.split(alive, p)) == 1 for p in deps):
                out.append(v)
        return out or cands          # never block every question

    def _priors(self, st, alive, k):
        b = st.belief
        acts = [SUBMIT]
        if k > 0:
            acts += [Action("ask", s) for s in self._legal_asks(st, alive)]
            if st.can_show(alive):
                acts.append(SHOW)
        if self.prior_mode == "uniform":
            return {a: 1.0 / len(acts) for a in acts}
        if self.prior_mode == "shapley":
            # Shapley-adjusted one-step values: ask v ~ acc + (1-eps)*phi_v - lam, submit ~ acc; softmax, small temperature
            cands = [a.slot for a in acts if a.kind == "ask"]
            phi = shapley_values(b, alive, cands) if cands else {}
            acc = b.acc_stop(alive)
            q = {a: (acc if a.kind != "ask" else acc + (1 - st.noise) * phi[a.slot] - st.lam) for a in acts}
            mx = max(q.values())
            e = {a: math.exp((q[a] - mx) / 0.02) for a in acts}
            z = sum(e.values())
            return {a: 0.05 / len(acts) + 0.95 * e[a] / z for a in acts}
        if self.prior_mode == "lookahead":
            # value of "take this action, then continue with the importance-ordered greedy policy",
            # computed exactly on the belief; softmax with a small temperature
            q = {a: self._greedy_q(st, alive, k, a) for a in acts}
            mx = max(q.values())
            e = {a: math.exp((q[a] - mx) / 0.02) for a in acts}
            z = sum(e.values())
            return {a: 0.05 / len(acts) + 0.95 * e[a] / z for a in acts}
        # Graph-node importance in one currency (EC2 weight removed): a question about slot v removes
        # gain(v) unless the user answers "no preference"; showing the current result is a test that
        # splits the belief into "this result" vs "not this result". Submitting gets the probability
        # that the current answer is already right; the rest is spread over tests by their gains.
        W = b.W(alive)
        p_now = b.acc_stop(alive)
        g = {}
        for a in acts[1:]:
            if a.kind == "ask":
                g[a] = (1 - st.noise) * b.gain(alive, a.slot)
            else:
                rest = b.after_show(alive, b.map_class(alive), False)
                tot = b.mass(alive)
                g[a] = W - (b.mass(rest) / tot) * b.W(rest) if rest else W
        z = sum(g.values())
        pri = {SUBMIT: p_now}
        for a, v in g.items():
            pri[a] = (1 - p_now) * (v / z if z > 0 else 1.0 / len(g))
        return {a: 0.05 / len(acts) + 0.95 * pri[a] for a in acts}

    def _greedy_v(self, st, alive, k, d=0):
        b = st.belief
        if self.la_depth is not None and d >= self.la_depth:
            return b.acc_stop(alive)
        key = ("gv", alive, k, d if self.la_depth is not None else 0)
        c = self._gcache
        if key in c:
            return c[key]
        a = self.greedy.act(st, alive, st.left(0) - k) if k > 0 else SUBMIT
        v = b.acc_stop(alive) if a.kind == "submit" else self._greedy_q(st, alive, k, a, d)
        c[key] = v
        return v

    def _greedy_q(self, st, alive, k, a, d=0):
        b = st.belief
        if a.kind == "submit" or k == 0:
            return b.acc_stop(alive)
        if a.kind == "show":
            p = b.acc_stop(alive)
            rest = b.after_show(alive, b.map_class(alive), False)
            return -st.show_cost + p + (1 - p) * (self._greedy_v(st, rest, k - 1, d + 1) if rest else 0.0)
        tot = b.mass(alive)
        ans = sum(b.mass(h) / tot * self._greedy_v(st, h, k - 1, d + 1) for h in b.split(alive, a.slot).values())
        return -st.lam + st.noise * self._greedy_v(st, alive, k - 1, d + 1) + (1 - st.noise) * ans

    # -- simulation ---------------------------------------------------------------------------
    def _sample_truth(self, st, alive):
        b = st.belief
        r, acc = self.rng.random() * b.mass(alive), 0.0
        for i in alive:
            acc += b.w[i]
            if acc >= r:
                return i
        return alive[-1]

    def _step(self, st, alive, a, truth):
        """Next alive set and the interaction cost."""
        b = st.belief
        if a.kind == "show":
            c = b.map_class(alive)
            return b.after_show(alive, c, b.cls[truth] == c), st.show_cost
        if self.rng.random() < st.noise:
            return alive, st.lam
        return b.observe(alive, a.slot, b.value(truth, a.slot)), st.lam

    def _final(self, st, alive, truth):
        if self.expected:
            return st.belief.acc_stop(alive)
        return 1.0 if st.belief.cls[truth] == st.belief.map_class(alive) else 0.0

    def _random_act(self, st, alive, k):
        if k == 0 or st.belief.certain(alive):
            return SUBMIT
        acts = [SUBMIT] + [Action("ask", s) for s in st.candidates(alive)] + ([SHOW] if st.can_show(alive) else [])
        return self.rng.choice(acts)

    def _order_act(self, st, alive, k):
        """Importance order without greedy's stopping rule: ask the most important slot until certain."""
        b = st.belief
        if k == 0 or b.certain(alive):
            return SUBMIT
        cands = st.candidates(alive)
        if not cands:
            return SUBMIT
        return Action("ask", max(cands, key=lambda v: (b.gain(alive, v), -st.askable.index(v))))

    def _topo_act(self, st, alive, k):
        """Closing-graph topological order (optimal on gated forests, RESULTS.md theory check 1), asked
        until the class is certain; stopping is left to the tree."""
        b = st.belief
        if k == 0 or b.certain(alive):
            return SUBMIT
        cands = st.candidates(alive)
        if not cands:
            return SUBMIT
        roots = b.closing_roots(alive, cands) or cands
        return Action("ask", max(roots, key=lambda v: (b.gain(alive, v), -st.askable.index(v))))

    def _rollout(self, st, alive, k, truth, traj):
        total = 0.0
        while True:
            if self.rollout == "random":
                a = self._random_act(st, alive, k)
            elif self.rollout == "topo":
                a = self._topo_act(st, alive, k)
            elif self.rollout == "order":
                a = self._order_act(st, alive, k)
            elif self.rollout == "shapley":         # ask in Shapley order until certain
                cands = st.candidates(alive)
                if k == 0 or st.belief.certain(alive) or not cands:
                    a = SUBMIT
                else:
                    phi = shapley_values(st.belief, alive, cands)
                    a = Action("ask", max(cands, key=lambda s: (phi[s], st.belief.gain(alive, s))))
            elif self.rollout == "fixed":           # ablation: no node importance, fixed slot order
                cands = st.candidates(alive)
                a = SUBMIT if k == 0 or st.belief.certain(alive) or not cands else Action("ask", cands[0])
            else:
                a = self.greedy.act(st, alive, st.left(0) - k) if k > 0 else SUBMIT
            if a.kind == "submit":
                return total + self._final(st, alive, truth)
            traj.append(a)
            alive, cost = self._step(st, alive, a, truth)
            total -= cost
            k -= 1

    def _score(self, node, a, sq):
        n, w = node.stats[a]
        # first-play urgency: an untried action starts at the node's running mean, not at 0
        q = w / n if n else (node.v / node.n if node.n else 0.5)
        if self.rave and a in self._rave and self._rave[a][0]:
            rn, rw = self._rave[a]
            beta = math.sqrt(self.rave_k / (3 * n + self.rave_k))
            q = (1 - beta) * q + beta * rw / rn
        return q + self.c * node.prior[a] * sq / (1 + n)

    def _simulate(self, st, tree, alive, k, truth, hist, traj):
        b = st.belief
        if k == 0 or b.certain(alive):
            return self._final(st, alive, truth)
        key = self._key(st, alive, k, hist)
        node = tree.get(key)
        if node is None:
            node = tree[key] = _Node(prior=self._priors(st, alive, k))
            for a in node.prior:
                node.stats[a] = [0, 0.0]
            r = self._rollout(st, alive, k, truth, traj)
            node.n, node.v = 1, r
            return r
        sq = math.sqrt(node.n + 1)
        a = max(node.stats, key=lambda x: self._score(node, x, sq))
        traj.append(a)
        if a.kind == "submit":
            r = self._final(st, alive, truth)
        else:
            nxt, cost = self._step(st, alive, a, truth)
            obs = len(nxt)
            r = -cost + self._simulate(st, tree, nxt, k - 1, truth, hist + ((a, obs),), traj)
        node.n += 1
        node.v += r
        node.stats[a][0] += 1
        node.stats[a][1] += r
        return r

    def _key(self, st, alive, k, hist):
        if self.abstract:
            return (st.belief.canon(alive), k)
        return (alive, k) if self.transpose else hist

    def _planning_setting(self, st):
        if self.plan_temp != 1.0:
            b = st.belief
            key = ("temp", id(b))
            if key not in self._shadow:
                self._shadow = {key: Belief(b.topo, b.hyps, prior=[max(x, 1e-300) ** self.plan_temp for x in b.w])}
            return Setting(self._shadow[key], st.askable, st.budget, st.lam, st.noise, st.show_cost)
        if self.classes:
            return st
        b = st.belief
        key = id(b)
        if key not in self._shadow:
            # same readings, but every reading is its own "result": the planner cannot see which
            # readings collapse to the same OSM result
            self._shadow = {key: Belief(b.topo, b.hyps, prior=b.w, threshold=1.01)}
        return Setting(self._shadow[key], st.askable, st.budget, st.lam, st.noise, st.show_cost)

    def act(self, st, alive, used):
        st = self._planning_setting(st)       # the no-topology planner only sees readings
        k = st.left(used)
        if k == 0 or st.belief.certain(alive) or not (st.candidates(alive) or st.can_show(alive)):
            return SUBMIT
        tree, self._rave, self._gcache = {}, {}, {}
        root_key = self._key(st, alive, k, ())
        for _ in range(self.n_sim):
            traj = []
            r = self._simulate(st, tree, alive, k, self._sample_truth(st, alive), (), traj)
            if self.rave:
                for a in set(traj):
                    e = self._rave.setdefault(a, [0, 0.0])
                    e[0] += 1
                    e[1] += r
        root = tree[root_key]
        return max(root.stats, key=lambda a: (root.stats[a][0], a.kind == "submit"))


def TopoMCTS(n_sim: int = 400, **kw):
    """MCTS on the decision problem the ambiguity topology defines: states = what is settled in the
    graph (merged into a DAG), actions = graph nodes (+ show), reward = result classes from executed
    readings, and simulated dialogues that ask in the topology's importance order until the result
    class is certain. The importance is used in rollouts, not as a prior: as a prior it re-imports
    greedy's myopia (out/mcts_sweep_*.jsonl, variant "+importance prior"). The rollout carries no
    one-step stopping rule ("order"): greedy's rule ("greedy" rollout, the 10-05 default) made TopoMCTS
    collapse to greedy on larger gated topologies (out/why_mcts_r300_m*.json) while real-data results
    are identical (out/rollout_check.json). Edge-constrained actions (frontier=True) did not help in
    the ablation and are off by default."""
    opts = dict(prior="uniform", rollout="order", frontier=False, transpose=True, classes=True, rave=False)
    opts.update(kw)
    return MCTS(n_sim=n_sim, name=kw.pop("name", "topo_mcts"), **{k: v for k, v in opts.items() if k != "name"})


def PlainMCTS(n_sim: int = 400, **kw):
    """MCTS without the topology (VisInteract-style): plans over raw readings without knowing which
    return the same result, random rollouts, history-keyed tree, no edge constraints."""
    opts = dict(prior="uniform", rollout="random", frontier=False, transpose=False, classes=False, rave=False)
    opts.update(kw)
    return MCTS(n_sim=n_sim, name="plain_mcts", **opts)


POLICIES = {p.name: p for p in (NoAsk, AskAll, Natural, Greedy, EIGReadings, ShowLoop, Optimal, MCTS)}
POLICIES.update(topo_mcts=TopoMCTS, plain_mcts=PlainMCTS)
