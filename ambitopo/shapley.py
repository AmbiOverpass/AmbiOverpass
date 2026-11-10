"""Execution-class coalition values and experimental search guides.

No gold intent enters this module. Values integrate over the current belief.
Noise is the existing independent 'no preference' erasure model, NOT answer errors.
The original accuracy-minus-cost MCTS return and legal action set are preserved.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

from .planners import Action, MCTS, SUBMIT


class CoalitionGame:
    """Exact small-slot game; each coalition asks its members once then submits.

    Coalition values average over answers and erasures. They are feasible
    nonadaptive gains, not optimal adaptive values. Enumerating all masks costs
    O(2**m * H * m); intended for the current M3/M4 (small m) only.
    """
    def __init__(self, belief, alive, slots, noise=0.0, max_slots=10, max_size=None):
        if not 0 <= noise <= 1:
            raise ValueError("noise must be in [0, 1]")
        self.b, self.alive = belief, tuple(alive)
        self.slots, self.noise = tuple(slots), noise
        if len(set(slots)) != len(slots) or any(v not in belief.slots for v in slots):
            raise ValueError("coalition slots must be unique belief slots")
        m = len(slots)
        self.max_size = m if max_size is None else min(m, max(0, max_size))
        if m > max_slots:
            raise ValueError("exact coalition prototype supports at most %d slots" % max_slots)
        self.now = belief.acc_stop(self.alive)
        tot = belief.mass(self.alive)
        cols = [belief.slots.index(v) for v in slots]
        self.raw = [0.0] * (1 << m)
        for mask in range(1, 1 << m):
            if mask.bit_count() > self.max_size:
                continue
            ix = [cols[j] for j in range(m) if mask & (1 << j)]
            groups = defaultdict(lambda: defaultdict(float))
            for h in self.alive:
                key = tuple(belief.hyps[h].assign[j] for j in ix)
                groups[key][belief.cls[h]] += belief.w[h]
            self.raw[mask] = max(0.0, sum(max(cm.values()) for cm in groups.values()) / tot - self.now)
        # Weighted subset transform: successful observations T subset S have
        # probability (1-noise)**|T| * noise**(|S|-|T|).
        self.gain = self.raw.copy()
        for j in range(m):
            bit = 1 << j
            for mask in range(1 << m):
                if mask & bit and mask.bit_count() <= self.max_size:
                    self.gain[mask] = (1 - noise) * self.gain[mask] + noise * self.gain[mask ^ bit]

    def attribution(self, budget=None):
        """Standard Shapley when budget >= m; a truncated semivalue otherwise.

        Average marginal gain over uniformly sampled predecessor sizes 0..k-1,
        then uniformly sampled subsets of that size. The truncated score is NOT
        a standard Shapley value and need not satisfy efficiency.
        """
        m = len(self.slots)
        k = m if budget is None else min(m, max(0, budget))
        if k > self.max_size:
            raise ValueError("requested attribution exceeds evaluated coalition size")
        out = {v: 0.0 for v in self.slots}
        if not k:
            return out
        for j, v in enumerate(self.slots):
            bit = 1 << j
            for mask in range(1 << m):
                r = mask.bit_count()
                if not mask & bit and r < k:
                    out[v] += (self.gain[mask | bit] - self.gain[mask]) / (k * math.comb(m - 1, r))
        return out

    def best(self, budget, cost):
        """Return a budget-feasible positive-net coalition, or the empty set."""
        if min(budget, len(self.slots)) > self.max_size:
            raise ValueError("requested coalition exceeds evaluated coalition size")
        best_mask, best_net = 0, 0.0
        for mask, gain in enumerate(self.gain):
            n = mask.bit_count()
            if n <= budget:
                net = gain - cost * n
                if net > best_net + 1e-12 or (abs(net - best_net) < 1e-12 and n < best_mask.bit_count()):
                    best_mask, best_net = mask, net
        return tuple(v for j, v in enumerate(self.slots) if best_mask & (1 << j)), best_net

    def member_scores(self, budget, cost):
        if min(budget, len(self.slots)) > self.max_size:
            raise ValueError("requested coalition exceeds evaluated coalition size")
        out = {v: -math.inf for v in self.slots}
        for mask, gain in enumerate(self.gain):
            n = mask.bit_count()
            if 0 < n <= budget:
                net = gain - cost * n
                for j, v in enumerate(self.slots):
                    if mask & (1 << j):
                        out[v] = max(out[v], net)
        return out


class SearchMCTS(MCTS):
    """Original MCTS with optional per-decision wall-time instrumentation.

    Deadline checked BETWEEN complete simulations, including initialization
    and coalition calculations. An atomic simulation may overshoot; record it.
    No implementation claims a hard real-time guarantee.
    """
    def __init__(self, *args, time_limit_ms=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_sim < 1:
            raise ValueError("n_sim must be positive")
        if time_limit_ms is not None and time_limit_ms <= 0:
            raise ValueError("time_limit_ms must be positive")
        self.time_limit_ms = time_limit_ms
        self.decisions = []

    def act(self, st, alive, used):
        start = time.perf_counter()
        st = self._planning_setting(st)
        k = st.left(used)
        n = 0
        action = SUBMIT
        if k and not st.belief.certain(alive) and (st.candidates(alive) or st.can_show(alive)):
            tree, self._rave, self._gcache = {}, {}, {}
            root_key = self._key(st, alive, k, ())
            # A positive number of simulations is required to initialize root.
            while n < self.n_sim:
                if n and self.time_limit_ms is not None and (time.perf_counter() - start) * 1000 >= self.time_limit_ms:
                    break
                traj = []
                r = self._simulate(st, tree, alive, k, self._sample_truth(st, alive), (), traj)
                if self.rave:
                    for a in set(traj):
                        e = self._rave.setdefault(a, [0, 0.0])
                        e[0] += 1
                        e[1] += r
                n += 1
            root = tree[root_key]
            action = max(root.stats, key=lambda a: (root.stats[a][0], a.kind == "submit"))
        elapsed = (time.perf_counter() - start) * 1000
        self.decisions.append({"ms": elapsed, "simulations": n,
                               "overshoot_ms": max(0.0, elapsed - self.time_limit_ms) if self.time_limit_ms else 0.0})
        return action


class ShapleyMCTS(SearchMCTS):
    """Ablatable prior/rollout guides; search return and actions stay unchanged.

    guide='shapley': budget-truncated attribution minus per-question cost.
    guide='coalition': best feasible net coalition containing each action.
    guided_prior=False keeps the original lookahead prior for rollout ablation.
    """
    def __init__(self, *args, guide="shapley", guided_prior=True,
                 guided_rollout=False, **kwargs):
        if guide not in ("shapley", "coalition"):
            raise ValueError("unknown guide")
        kwargs.setdefault("prior", "lookahead")
        kwargs.setdefault("rollout", "order")
        super().__init__(*args, **kwargs)
        self.guide, self.guided_prior, self.guided_rollout = guide, guided_prior, guided_rollout
        self._games = {}
        self._game_belief = None

    def _game(self, st, alive, k):
        if st.show_cost is not None or self.frontier:
            raise ValueError("prototype supports unrestricted ask-only M3/M4 protocol")
        if self._game_belief is not st.belief:
            self._games.clear()
            self._game_belief = st.belief
        slots = tuple(st.candidates(alive))
        cap = min(k, len(slots))
        key = (tuple(alive), slots, st.noise, cap)
        if key not in self._games:
            self._games[key] = CoalitionGame(st.belief, alive, slots, st.noise, max_size=cap)
        return self._games[key]

    def _priors(self, st, alive, k):
        if not self.guided_prior:
            return super()._priors(st, alive, k)
        if k <= 0:
            return {SUBMIT: 1.0}
        game = self._game(st, alive, k)
        if self.guide == "shapley":
            scores = {v: x - st.lam for v, x in game.attribution(k).items()}
        else:
            scores = game.member_scores(k, st.lam)
        q = {SUBMIT: 0.0, **{Action("ask", v): x for v, x in scores.items()}}
        mx = max(q.values())
        e = {a: math.exp((x - mx) / 0.02) for a, x in q.items()}
        z = sum(e.values())
        return {a: 0.05 / len(q) + 0.95 * x / z for a, x in e.items()}

    def _rollout(self, st, alive, k, truth, traj):
        if not self.guided_rollout:
            return super()._rollout(st, alive, k, truth, traj)
        total = 0.0
        while k > 0 and not st.belief.certain(alive):
            game = self._game(st, alive, k)
            if not game.slots:
                break
            phi = game.attribution(k)
            if self.guide == "coalition":
                members, net = game.best(k, st.lam)
                if not members or net <= 1e-12:
                    break   # rollout choice only: tree still explores all asks
                slot = max(members, key=lambda v: (phi[v], -game.slots.index(v)))
            else:
                # Attribution ranks questions; feasible coalition gain decides
                # whether this rollout continues. Never use phi > cost to stop.
                members, net = game.best(k, st.lam)
                if not members or net <= 1e-12:
                    break
                slot = max(game.slots, key=lambda v: (phi[v], -game.slots.index(v)))
            a = Action("ask", slot)
            traj.append(a)
            alive, cost = self._step(st, alive, a, truth)
            total -= cost
            k -= 1
        return total + self._final(st, alive, truth)
