"""Tune the judgment-call dials on the five DFL tuning matches, one dial at a time.

    python tune.py

Each dial is tried at every listed value with the others fixed, the best is kept, and the whole
list is passed over twice. The objective is the official spellF1, averaged over both versions of
the truth and every way of hiding the ball, so that no setting wins by suiting one condition. The
two held-out matches are never read here, and neither is the competition match.
"""

import multiprocessing
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
from possession import config, decode, evidence, external, features  # noqa: E402

SEEDS = (0,)
PASSES = 2
WORKERS = 8

# A dial only moves if it gains more than this: smaller gains are noise, and chasing them would
# fit the tuning matches rather than football. The change of team is judged in tools/proof_table.py.
NOISE_FLOOR = 0.002

# Dials that change the facts, so features must be recomputed. The rest only change the votes.
FEATURE_DIALS = {"SLOW_CUT_SPEED", "SLOW_SCALE", "NEAR_RADIUS", "OUT_MARGIN", "CONTROL_MARGIN"}

DIALS = (
    ("P_TO_DEAD", (2, 4, 6, 8, 12, 16, 24)),
    ("P_FROM_DEAD", (1, 2, 4, 6, 8, 12)),
    ("W_SLOW", (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)),
    ("SLOW_CUT_SPEED", (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5)),
    ("SLOW_SCALE", (0.2, 0.4, 0.8)),
    ("W_NEAR", (0.0, 0.05, 0.1, 0.2, 0.4)),
    ("W_MOVING", (0.0, 0.1, 0.25, 0.5, 1.0)),
    ("W_OUT", (0.5, 1.0, 2.0, 4.0)),
    ("NEAR_RADIUS", (2.0, 3.0, 5.0)),
    ("CONTROL_MARGIN", (0.0, 0.25, 0.5, 1.0)),
    ("OUT_MARGIN", (0.11, 0.25, 0.5)),
)


CASES, FACTS = [], []


def _compute_facts(index):
    return features.compute(CASES[index][2].tracking)


def _score_case(index):
    case = CASES[index]
    labels = decode.decode(evidence.build(FACTS[index], []), case[2].tracking.halves)
    results = [bench.score(labels, truth) for truth in case[3:5]]
    return [r["spell"] for r in results], [r["macro"] for r in results]


def _parallel(function):
    # A fresh forked pool each time, so the workers see the dial values just set in this process.
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        return pool.map(function, range(len(CASES)))


def refresh_facts():
    FACTS[:] = _parallel(_compute_facts)


def objective():
    results = _parallel(_score_case)
    return (float(np.mean([s for spells, _ in results for s in spells])),
            float(np.mean([m for _, macros in results for m in macros])))


def main():
    for seed in SEEDS:
        CASES.extend(bench.versions(external.TRAIN_MATCHES, bench.HIDDEN_SCENARIOS, seed))
    print("{} cases, proof fixed at {:g}".format(len(CASES), config.P_TEAM_SWITCH), flush=True)
    refresh_facts()

    best = objective()
    print("start  spellF1 {:.4f}  macroF1 {:.4f}".format(*best), flush=True)

    for round_number in range(1, PASSES + 1):
        for name, values in DIALS:
            kept = getattr(config, name)
            outcomes = []
            for value in values:
                setattr(config, name, value)
                if name in FEATURE_DIALS:
                    refresh_facts()
                outcomes.append((objective(), value))
            winner = max(outcomes, key=lambda item: item[0])
            if winner[0][0] < best[0] + NOISE_FLOOR:
                winner = (best, kept)  # within the noise: keep what we had
            best = winner[0]
            setattr(config, name, winner[1])
            if name in FEATURE_DIALS:
                refresh_facts()
            print("pass {}  {:<16} {:<7g} spellF1 {:.4f}  macroF1 {:.4f}   [{}]".format(
                round_number, name, winner[1], best[0], best[1],
                "  ".join("{:g}:{:.3f}".format(v, r[0]) for r, v in outcomes)), flush=True)

    print("\nFINAL  spellF1 {:.4f}  macroF1 {:.4f}".format(*best))
    for name, _ in DIALS:
        print("{} = {:g}".format(name, getattr(config, name)))


if __name__ == "__main__":
    main()
