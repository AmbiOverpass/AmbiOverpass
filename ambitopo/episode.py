"""Simulated user and the interaction loop."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .belief import Belief, jaccard
from .planners import Policy, Setting


class SimulatedUser:
    """Knows the true joint reading. Answers a question about a slot with the true reading, or
    with "no preference" (None) with probability `noise`. It never volunteers anything else."""

    def __init__(self, belief: Belief, truth: int, noise: float = 0.0, seed: int = 0):
        self.b = belief
        self.truth = truth
        self.noise = noise
        self.rng = random.Random(seed)

    def answer(self, slot: str):
        if self.rng.random() < self.noise:
            return None
        return self.b.value(self.truth, slot)

    def judge(self, cls: int) -> bool:
        """Shown a result: is this what I meant? (taken as reliable)"""
        return self.b.cls[self.truth] == cls


class AnswerTableUser(SimulatedUser):
    """V4 robustness check: answers come from a fixed table (an LLM that read the precise request picked one value per
    slot); "no preference" with probability `noise` as before; a slot missing from the table or unparsed counts as no
    preference."""

    def __init__(self, belief: Belief, truth: int, answers: dict, noise: float = 0.0, seed: int = 0):
        super().__init__(belief, truth, noise, seed)
        self.answers = answers

    def answer(self, slot: str):
        if self.rng.random() < self.noise:
            return None
        return self.answers.get(slot)


@dataclass
class EpisodeResult:
    policy: str
    truth: int
    submitted: int                 # index of the submitted (representative) hypothesis
    success: bool                  # same result class as the truth
    jaccard: float                 # overlap of submitted vs true result set
    asks: int
    shows: int = 0
    wasted: int = 0                # asks about a slot that could not change the result class
    missed: bool = False           # wrong result while a useful question was still possible
    trace: list = field(default_factory=list)


def run_episode(st: Setting, policy: Policy, user: SimulatedUser, max_steps: int = 50) -> EpisodeResult:
    b = st.belief
    alive, asks, shows, wasted, trace = b.root, 0, 0, 0, []
    for _ in range(max_steps):
        a = policy.act(st, alive, asks + shows)
        if a.kind == "submit":
            break
        if a.kind == "show":
            c = b.map_class(alive)
            ok = user.judge(c)
            alive = b.after_show(alive, c, ok)
            shows += 1
            trace.append(("show", ok, len(alive)))
            continue
        if b.gain(alive, a.slot) <= 1e-12:
            wasted += 1
        ans = user.answer(a.slot)
        asks += 1
        if ans is not None:
            nxt = b.observe(alive, a.slot, ans)
            if nxt:                      # an answer that contradicts every remaining reading is discarded
                alive = nxt
        trace.append((str(a), ans, len(alive)))
    cls = b.map_class(alive)
    rep = b.representative(alive, cls)
    ok = cls == b.cls[user.truth]
    missed = (not ok) and any(b.gain(alive, v) > 1e-12 for v in st.candidates(alive))
    return EpisodeResult(policy.name, user.truth, rep, ok, jaccard(b.hyps[rep].result, b.hyps[user.truth].result),
                         asks, shows, wasted, missed, trace)
