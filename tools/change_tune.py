"""Tune the change detector on the DFL tuning matches, then judge it on every other match.

    python train.py --candidate v12 && python tools/change_tune.py v12

All predictions are from models that never saw the match. The two settings are chosen on the five
DFL tuning matches only; the two other DFL matches and Metrica's three are kept for judging. The
comparison is the same models with and without the detector, so only the detector differs.
"""

import glob
import multiprocessing
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
from possession import config, decode, external, model, tags  # noqa: E402

PM = bench.PM
TOLLS = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
DISCOUNTS = (0.0, 0.5, 0.75, 0.9, 0.97)
CASES = []
MATCHES = {}


def labels_for(index, toll, discount):
    name, case = CASES[index]
    match_id = name.split("_")[0]
    halves = tuple(tuple(int(v) for v in row) for row in case["halves"])
    evidence = model.votes(case["p_home"], case["p_dead"], case["out"], cue=case["cue"])
    evidence = tags.add_votes(evidence, tags.imitate(MATCHES[match_id], zlib.crc32(name.encode())),
                              config.TAG_WEIGHT_DEAD, config.TAG_LOOKBACK_S, config.TAG_WEIGHT_TEAM)
    cost = decode.transition_costs(toll, config.V2_P_TO_DEAD, config.V2_P_FROM_DEAD)
    team_cost = np.maximum(toll * (1 - discount * case["p_change"]), 0.25)
    return decode.decode(evidence, halves, cost, team_cost)


def _score(job):
    index, toll, discount = job
    name, case = CASES[index]
    labels = labels_for(index, toll, discount)
    gt = PM.label_codes(case["truth"])
    runs = PM.valid_runs(gt)
    filled = PM.fill_abstentions(PM.label_codes(labels), runs)
    gt_spells = PM.possession_spells(gt, runs)
    pred_spells = PM.possession_spells(filled, runs)
    found = int(PM.matched_gt_flags(PM.spell_overlaps(gt_spells, pred_spells), len(gt_spells),
                                    0.5).sum())
    raw = bench.score(labels, case["truth"])
    return (raw["spell"], bench.score(labels, case["touchless"])["spell"], raw["macro"],
            len(pred_spells), found, len(gt_spells))


def mean_over(indices, toll, discount, pool):
    return np.mean(pool.map(_score, [(i, toll, discount) for i in indices]), axis=0)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "v12"
    for path in sorted(glob.glob(os.path.join(config.ROOT, "reports", "oof_" + name, "*.npz"))):
        file_name = os.path.basename(path)
        CASES.append((file_name, dict(np.load(path))))
        match_id = file_name.split("_")[0]
        if match_id not in MATCHES:
            MATCHES[match_id] = external.load_match(match_id)
    tuning = [i for i, (n, _) in enumerate(CASES) if n.split("_")[0] in external.TRAIN_MATCHES]
    judged = {"DFL held-out": [i for i, (n, _) in enumerate(CASES)
                               if n.split("_")[0] in external.TEST_MATCHES],
              "Metrica": [i for i, (n, _) in enumerate(CASES)
                          if n.split("_")[0] in external.METRICA_MATCHES]}

    with multiprocessing.get_context("fork").Pool(8) as pool:
        print("Tuning on {} cases from the five DFL tuning matches".format(len(tuning)))
        print("{:>6}{:>10}{:>10}{:>11}{:>10}".format("toll", "discount", "spellF1", "no-touches",
                                                     "possess"))
        best, best_row = (config.V2_P_TEAM_SWITCH, 0.0), None
        for toll in TOLLS:
            for discount in DISCOUNTS:
                row = mean_over(tuning, toll, discount, pool)
                print("{:>6g}{:>10g}{:>10.3f}{:>11.3f}{:>10.0f}".format(toll, discount, row[0], row[1],
                                                                        row[3]))
                if best_row is None or (row[0] + row[1]) > (best_row[0] + best_row[1]):
                    best, best_row = (toll, discount), row
        print("\nBEST toll {:g}, discount {:g}".format(*best))

        print("\nJudged on matches the settings never saw (spellF1, macroF1, reported, found, true):")
        for group, indices in judged.items():
            for label, setting in (("without detector", (config.V2_P_TEAM_SWITCH, 0.0)),
                                   ("with detector", best)):
                row = mean_over(indices, *setting, pool)
                print("  {:<14}{:<18} spellF1 {:.3f}  no-touches {:.3f}  macroF1 {:.3f}  "
                      "reported {:.0f}  found {:.0f} of {:.0f}  fake {:.0f}".format(
                          group, label, row[0], row[1], row[2], row[3], row[4], row[5],
                          row[3] - row[4]))


if __name__ == "__main__":
    main()
