"""Tune the decoder for version 2 on predictions for left-out DFL matches.

    python train.py && python tune_v2.py

Every prediction here comes from models that never saw that match, so the score is a fair guide.
The objective is the official spellF1, averaged over both versions of the truth and every way of
hiding the ball. A dial only moves if it gains more than the noise.
"""

import glob
import multiprocessing
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
from possession import config, decode, model  # noqa: E402

WORKERS = 8
PASSES = 2
NOISE_FLOOR = 0.002
OOF_DIR = os.path.join(config.ROOT, "reports", "oof")

DIALS = (
    ("V2_P_TEAM_SWITCH", (2, 4, 6, 8, 12, 16, 24, 32, 48, 64)),
    ("V2_ALPHA_DEAD", (0.1, 0.25, 0.5, 1.0, 2.0)),
    ("V2_P_TO_DEAD", (2, 4, 8, 12, 16, 24, 32)),
    ("V2_P_FROM_DEAD", (2, 4, 8, 12, 16, 24, 32)),
    ("V2_W_OUT", (0.0, 1.0, 2.0, 4.0, 8.0)),
)

CASES = []


def _score(index):
    case = CASES[index]
    halves = tuple(tuple(int(v) for v in row) for row in case["halves"])
    evidence = model.votes(case["p_home"], case["p_dead"], case["out"])
    cost = decode.transition_costs(config.V2_P_TEAM_SWITCH, config.V2_P_TO_DEAD,
                                   config.V2_P_FROM_DEAD)
    labels = decode.decode(evidence, halves, cost)
    results = [bench.score(labels, case[key]) for key in ("truth", "touchless")]
    return [r["spell"] for r in results], [r["macro"] for r in results], results[0]["possessions"]


def objective():
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        results = pool.map(_score, range(len(CASES)))
    return (float(np.mean([s for spells, _, _ in results for s in spells])),
            float(np.mean([m for _, macros, _ in results for m in macros])),
            float(np.mean([p for _, _, p in results])))


def main():
    folder = OOF_DIR + ("_" + sys.argv[1] if len(sys.argv) > 1 else "")
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        CASES.append(dict(np.load(path)))
    print("{} left-out cases".format(len(CASES)), flush=True)

    best = objective()
    print("start  spellF1 {:.4f}  macroF1 {:.4f}  possessions {:.0f}".format(*best), flush=True)
    for round_number in range(1, PASSES + 1):
        for name, values in DIALS:
            kept = getattr(config, name)
            outcomes = []
            for value in values:
                setattr(config, name, value)
                outcomes.append((objective(), value))
            winner = max(outcomes, key=lambda item: item[0][0])
            if winner[0][0] < best[0] + NOISE_FLOOR:
                winner = (best, kept)
            best = winner[0]
            setattr(config, name, winner[1])
            print("pass {}  {:<17}{:<7g} spellF1 {:.4f}  macroF1 {:.4f}  poss {:.0f}   [{}]".format(
                round_number, name, winner[1], best[0], best[1], best[2],
                "  ".join("{:g}:{:.3f}".format(v, r[0]) for r, v in outcomes)), flush=True)

    print("\nFINAL  spellF1 {:.4f}  macroF1 {:.4f}  possessions {:.0f}".format(*best))
    for name, _ in DIALS:
        print("{} = {:g}".format(name, getattr(config, name)))


if __name__ == "__main__":
    main()
