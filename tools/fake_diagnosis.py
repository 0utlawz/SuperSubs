"""What are the possessions we report that match nothing real?

    python tools/fake_diagnosis.py v3

Uses predictions for DFL matches left out of training. Each reported possession that matches no true
possession at IoU 0.5 is put in one type, because each type needs a different fix.
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
OOF_DIR = os.path.join(config.ROOT, "reports", "oof")
TYPES = ("Mostly during a dead ball", "Wrong team", "Swallowed real possessions",
         "Fragment of a longer one", "Right idea, edges off")
CASES = []


def classify(start, stop, code, truth, gt, gt_spells):
    window = truth[start:stop]
    length = stop - start
    if np.mean(window == "D") > 0.5:
        return 0
    team = "H" if code == PM.H else "A"
    other = "A" if team == "H" else "H"
    if np.mean(window == other) > np.mean(window == team):
        return 1
    best, best_overlap = None, 0
    for gt_start, gt_stop, gt_code in gt_spells:
        if gt_code != code:
            continue
        overlap = min(stop, gt_stop) - max(start, gt_start)
        if overlap > best_overlap:
            best, best_overlap = (gt_start, gt_stop), overlap
    if best is None:
        return 1
    true_length = best[1] - best[0]
    if length > 1.5 * true_length:
        return 2
    if length < 0.67 * true_length:
        return 3
    return 4


def _run(index):
    case = CASES[index]
    halves = tuple(tuple(int(v) for v in row) for row in case["halves"])
    evidence = model.votes(case["p_home"], case["p_dead"], case["out"])
    cost = decode.transition_costs(config.V2_P_TEAM_SWITCH, config.V2_P_TO_DEAD,
                                   config.V2_P_FROM_DEAD)
    labels = decode.decode(evidence, halves, cost)
    truth = case["truth"]
    gt = PM.label_codes(truth)
    pred = PM.label_codes(labels)
    runs = PM.valid_runs(gt)
    filled = PM.fill_abstentions(pred, runs)
    gt_spells = PM.possession_spells(gt, runs)
    pred_spells = PM.possession_spells(filled, runs)
    overlaps = PM.spell_overlaps(gt_spells, pred_spells)

    matched = set()
    used = set()
    for iou, gt_index, pred_index in overlaps:
        if iou < THRESHOLD:
            break
        if gt_index in used or pred_index in matched:
            continue
        used.add(gt_index)
        matched.add(pred_index)

    fakes = [[] for _ in TYPES]
    for index_, (start, stop, code) in enumerate(pred_spells):
        if index_ not in matched:
            fakes[classify(start, stop, code, truth, gt, gt_spells)].append(
                (stop - start) / config.FPS)
    return len(pred_spells), len(matched), fakes


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    folder = OOF_DIR + ("_" + name if name else "")
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        CASES.append(dict(np.load(path)))
    with multiprocessing.get_context("fork").Pool(8) as pool:
        results = pool.map(_run, range(len(CASES)))

    n = len(results)
    print("reported {:.0f}, matched {:.0f}, fake {:.0f} per match".format(
        np.mean([r[0] for r in results]), np.mean([r[1] for r in results]),
        np.mean([r[0] - r[1] for r in results])))
    print("{:<30}{:>11}{:>9}{:>13}".format("type of fake", "per match", "share", "median length"))
    total = sum(len(x) for r in results for x in r[2])
    for position, name_ in enumerate(TYPES):
        lengths = [v for r in results for v in r[2][position]]
        print("{:<30}{:>11.0f}{:>8.0f}%{:>12.1f}s".format(
            name_, len(lengths) / n, 100 * len(lengths) / total,
            float(np.median(lengths)) if lengths else 0.0))


if __name__ == "__main__":
    main()
