"""Checks of the decoder on made-up evidence -- no match data involved.

    python tests/test_decode.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession.decode import transition_costs, viterbi  # noqa: E402
from possession.evidence import H, A, D  # noqa: E402

COST = transition_costs()
MIN_DEAD = 50


def evidence(n, spans):
    """(n, 3) evidence with weight w on state s over [start, end) for each (s, start, end, w)."""
    e = np.zeros((n, 3))
    for state, start, end, weight in spans:
        e[start:end, state] += weight
    return e


def runs_of(path):
    starts = np.flatnonzero(np.concatenate(([True], path[1:] != path[:-1])))
    ends = np.concatenate((starts[1:], [len(path)]))
    return [("HAD"[path[s]], int(s), int(e)) for s, e in zip(starts, ends)]


def test_short_dead_blip_is_ignored():
    # 10 frames of strong dead-ball votes inside a Barca possession: shorter than 2 s, so no stop.
    e = evidence(400, [(H, 0, 400, 0.5), (D, 200, 210, 3.0)])
    assert runs_of(viterbi(e, COST, MIN_DEAD)) == [("H", 0, 400)]


def test_real_stoppage_is_exact():
    e = evidence(400, [(H, 0, 150, 1.0), (D, 150, 250, 1.0), (A, 250, 400, 1.0)])
    assert runs_of(viterbi(e, COST, MIN_DEAD)) == [("H", 0, 150), ("D", 150, 250), ("A", 250, 400)]


def test_no_fake_stoppage_as_a_shortcut():
    # A 1 s Athletic takeover with slow players around it. With a 2 s floor the cheapest story is
    # a direct change of team, not a few frames of invented dead ball.
    e = evidence(300, [(H, 0, 100, 1.0), (D, 95, 105, 0.15), (A, 100, 300, 1.0)])
    labels = [state for state, _, _ in runs_of(viterbi(e, COST, MIN_DEAD))]
    assert labels == ["H", "A"]


def test_half_may_start_dead():
    e = evidence(300, [(D, 0, 20, 1.0), (A, 20, 300, 1.0)])
    assert runs_of(viterbi(e, COST, MIN_DEAD)) == [("D", 0, 20), ("A", 20, 300)]


def test_every_inner_stoppage_is_long_enough():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, (5000, 3))
    for state, start, end in runs_of(viterbi(e, COST, MIN_DEAD)):
        if state == "D" and start > 0 and end < 5000:
            assert end - start >= MIN_DEAD, (start, end)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok  ", name)
