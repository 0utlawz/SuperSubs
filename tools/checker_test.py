"""Train and test the possession checker on matches left out of training.

    python tools/checker_test.py v3

For each DFL tuning match, the checker is trained on the other four and scored on that one, so it
never judges a match it learned from. Reported figures compare with no checker at all.
"""

import glob
import multiprocessing
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
from possession import checker, config, decode, external, model  # noqa: E402

PM = bench.PM
OOF_DIR = os.path.join(config.ROOT, "reports", "oof")
THRESHOLDS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
IOU = 0.5
CASES = []


def matched(labels, truth):
    """Which reported possessions match a true one, plus the counts behind found and fake."""
    gt = PM.label_codes(truth)
    pred = PM.label_codes(labels)
    runs = PM.valid_runs(gt)
    filled = PM.fill_abstentions(pred, runs)
    gt_spells = PM.possession_spells(gt, runs)
    pred_spells = PM.possession_spells(filled, runs)
    overlaps = PM.spell_overlaps(gt_spells, pred_spells)
    used_gt, used_pred = set(), set()
    for iou, gt_index, pred_index in overlaps:
        if iou < IOU:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
    return pred_spells, used_pred


def prepare(index):
    path, case = CASES[index]
    halves = tuple(tuple(int(v) for v in row) for row in case["halves"])
    evidence = model.votes(case["p_home"], case["p_dead"], case["out"])
    cost = decode.transition_costs(config.V2_P_TEAM_SWITCH, config.V2_P_TO_DEAD,
                                   config.V2_P_FROM_DEAD)
    labels = decode.decode(evidence, halves, cost)
    spans, rows = checker.chunk_rows(labels, case["p_home"], case["p_dead"], case["out"], halves)
    pred_spells, hits = matched(labels, case["touchless"])
    starts = {spell[0]: i for i, spell in enumerate(pred_spells)}
    target = np.array([starts.get(span[0], -1) in hits for span in spans])
    return os.path.basename(path).split("_")[0], labels, spans, rows, target


def report(job):
    labels, truth, touchless = job
    spells, hits = matched(labels, truth)
    return (len(spells), len(hits), bench.score(labels, truth)["spell"],
            bench.score(labels, touchless)["spell"], bench.score(labels, truth)["macro"])


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "v3"
    for path in sorted(glob.glob(os.path.join(OOF_DIR + "_" + name, "*.npz"))):
        CASES.append((path, dict(np.load(path))))
    with multiprocessing.get_context("fork").Pool(8) as pool:
        prepared = pool.map(prepare, range(len(CASES)))

    scored = {threshold: [] for threshold in THRESHOLDS}
    for left_out in external.TRAIN_MATCHES:
        train = [p for p in prepared if p[0] != left_out]
        fitted = checker.make_model().fit(np.concatenate([p[3] for p in train]),
                                          np.concatenate([p[4] for p in train]))
        for index, entry in enumerate(prepared):
            if entry[0] != left_out:
                continue
            _, labels, spans, rows, _ = entry
            p_real = fitted.predict_proba(rows)[:, 1]
            case = CASES[index][1]
            for threshold in THRESHOLDS:
                cleaned = checker.remove(labels, spans, p_real, threshold)
                scored[threshold].append((cleaned, case["truth"], case["touchless"]))

    print("{} left-out cases; true possessions per match: {:.0f}".format(
        len(CASES), np.mean([len(matched(c[1]["truth"], c[1]["truth"])[0]) for c in CASES])))
    print("{:>10}{:>10}{:>8}{:>7}{:>10}{:>12}{:>9}".format(
        "threshold", "reported", "found", "fake", "spellF1", "no-touches", "macroF1"))
    with multiprocessing.get_context("fork").Pool(8) as pool:
        for threshold in THRESHOLDS:
            rows = pool.map(report, scored[threshold])
            reported = np.mean([r[0] for r in rows])
            found = np.mean([r[1] for r in rows])
            print("{:>10g}{:>10.0f}{:>8.0f}{:>7.0f}{:>10.3f}{:>12.3f}{:>9.3f}{}".format(
                threshold, reported, found, reported - found, np.mean([r[2] for r in rows]),
                np.mean([r[3] for r in rows]), np.mean([r[4] for r in rows]),
                "   <- no checker" if threshold == 0.0 else ""))

    final = checker.make_model().fit(np.concatenate([p[3] for p in prepared]),
                                     np.concatenate([p[4] for p in prepared]))
    path = os.path.join(config.ROOT, "models", "checker_{}.pkl".format(name))
    with open(path, "wb") as handle:
        pickle.dump(final, handle)


if __name__ == "__main__":
    main()
