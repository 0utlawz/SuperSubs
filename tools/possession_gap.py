"""Which possessions do we miss, and what would chasing the raw count cost?

    python tools/possession_gap.py

Uses predictions for DFL matches left out of training, so nothing here touches the held-out
matches, the Clasicos or the competition match. A true possession counts as found when one of ours
overlaps it with an IoU of at least 0.5, the middle threshold of the official metric.
"""

import glob
import multiprocessing
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
from possession import config, decode, model  # noqa: E402

PM = bench.PM
THRESHOLD = 0.5
TOLLS = (0.5, 1, 2, 3, 4, 6, 8, 12)
BUCKETS = ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 1e9))
OOF_DIR = os.path.join(config.ROOT, "reports", "oof")
CASES = []


def _run(job):
    index, toll = job
    case = CASES[index]
    halves = tuple(tuple(int(v) for v in row) for row in case["halves"])
    evidence = model.votes(case["p_home"], case["p_dead"], case["out"])
    cost = decode.transition_costs(toll, config.V2_P_TO_DEAD, config.V2_P_FROM_DEAD)
    labels = decode.decode(evidence, halves, cost)

    gt = PM.label_codes(case["truth"])
    pred = PM.label_codes(labels)
    runs = PM.valid_runs(gt)
    filled = PM.fill_abstentions(pred, runs)
    gt_spells = PM.possession_spells(gt, runs)
    pred_spells = PM.possession_spells(filled, runs)
    overlaps = PM.spell_overlaps(gt_spells, pred_spells)
    found = PM.matched_gt_flags(overlaps, len(gt_spells), THRESHOLD)
    tp, fp, fn = PM.match_spells(overlaps, len(gt_spells), len(pred_spells), THRESHOLD)
    seconds = np.array([(stop - start) / config.FPS for start, stop, _ in gt_spells])
    return (toll, len(gt_spells), len(pred_spells), tp, fp, fn, bench.score(labels, case["truth"])["spell"],
            bench.score(labels, case["touchless"])["spell"], seconds, np.asarray(found, dtype=bool))


def main():
    folder = OOF_DIR + ("_" + sys.argv[1] if len(sys.argv) > 1 else "")
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        CASES.append(dict(np.load(path)))
    jobs = [(i, toll) for i in range(len(CASES)) for toll in TOLLS]
    with multiprocessing.get_context("fork").Pool(8) as pool:
        results = pool.map(_run, jobs)

    print("{} left-out cases. What each toll does (IoU {:.1f}):".format(len(CASES), THRESHOLD))
    print("{:>6}{:>9}{:>9}{:>8}{:>8}{:>9}{:>9}{:>11}".format(
        "toll", "reported", "true", "found", "fake", "missed", "spellF1", "no-touches"))
    for toll in TOLLS:
        rows = [r for r in results if r[0] == toll]
        print("{:>6g}{:>9.0f}{:>9.0f}{:>8.0f}{:>8.0f}{:>9.0f}{:>9.3f}{:>11.3f}".format(
            toll, *[np.mean([r[k] for r in rows]) for k in (2, 1, 3, 4, 5, 6, 7)]))

    print("\nWhich true possessions we find, at today's toll of {:g}:".format(config.V2_P_TEAM_SWITCH))
    rows = [r for r in results if r[0] == config.V2_P_TEAM_SWITCH]
    seconds = np.concatenate([r[8] for r in rows])
    found = np.concatenate([r[9] for r in rows])
    per_case = len(rows)
    print("{:<12}{:>12}{:>9}{:>11}{:>12}".format("length", "per match", "share", "found", "missed/match"))
    for low, high in BUCKETS:
        band = (seconds >= low) & (seconds < high)
        print("{:<12}{:>12.0f}{:>8.0f}%{:>10.0f}%{:>12.0f}".format(
            "{:g}-{} s".format(low, "inf" if high > 1e8 else "{:g}".format(high)),
            band.sum() / per_case, 100 * band.mean(), 100 * found[band].mean(),
            (band & ~found).sum() / per_case))


if __name__ == "__main__":
    main()
