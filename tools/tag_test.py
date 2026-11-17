"""What are Smart Tagging's set-piece tags worth? Tune and test them on imitation tags.

    python tools/tag_test.py v3            # tune on matches left out of training
    python tools/tag_test.py v3 --test     # score the settings in config on the held-out matches

Real tags exist only for the competition match, where nothing may be tuned. So the DFL restarts are
turned into imitation tags (possession/tags.py). Each case gets its own fixed random tags.
"""

import glob
import multiprocessing
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
from possession import config, decode, external, features, model, tags  # noqa: E402

PM = bench.PM
OOF_DIR = os.path.join(config.ROOT, "reports", "oof")
CASES = []

WEIGHTS_DEAD = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
LOOKBACKS = (2.0, 4.0, 8.0, 15.0)
WEIGHTS_TEAM = (0.0, 0.25, 0.5, 1.0, 2.0)
# (name, false tags, timing error in seconds): how the tags hold up if they are worse than assumed.
QUALITY = (("as assumed", 0.1, 1.0), ("more false tags", 0.3, 1.0), ("worse timing", 0.1, 2.5),
           ("both", 0.3, 2.5))


def measure(labels, truth, touchless):
    gt = PM.label_codes(truth)
    pred = PM.label_codes(labels)
    runs = PM.valid_runs(gt)
    filled = PM.fill_abstentions(pred, runs)
    gt_spells = PM.possession_spells(gt, runs)
    pred_spells = PM.possession_spells(filled, runs)
    overlaps = PM.spell_overlaps(gt_spells, pred_spells)
    found = int(PM.matched_gt_flags(overlaps, len(gt_spells), 0.5).sum())
    raw = bench.score(labels, truth)
    return (len(pred_spells), found, raw["spell"], bench.score(labels, touchless)["spell"],
            raw["macro"], raw["stoppages"], raw["dead"])


def _case(job):
    index, weight_dead, lookback, weight_team = job
    case = CASES[index]
    evidence = tags.add_votes(case["votes"].copy(), case["tags"], weight_dead, lookback, weight_team)
    return measure(decode.decode(evidence, case["halves"], case["cost"]), case["truth"],
                   case["touchless"])


def run(jobs):
    with multiprocessing.get_context("fork").Pool(8) as pool:
        return np.mean(pool.map(_case, jobs), axis=0)


def show(label, row):
    print("{:<34}{:>9.0f}{:>7.0f}{:>6.0f}{:>9.3f}{:>11.3f}{:>9.3f}{:>7.0f}{:>7.1f}%".format(
        label, row[0], row[1], row[0] - row[1], row[2], row[3], row[4], row[5], 100 * row[6]))


HEADER = "{:<34}{:>9}{:>7}{:>6}{:>9}{:>11}{:>9}{:>7}{:>8}".format(
    "", "reported", "found", "fake", "spellF1", "no-touches", "macroF1", "stops", "dead")


def tune(name):
    cost = decode.transition_costs(config.V2_P_TEAM_SWITCH, config.V2_P_TO_DEAD,
                                   config.V2_P_FROM_DEAD)
    matches = {m: external.load_match(m) for m in external.TRAIN_MATCHES}
    for path in sorted(glob.glob(os.path.join(OOF_DIR + "_" + name, "*.npz"))):
        case = dict(np.load(path))
        file_name = os.path.basename(path)
        case["halves"] = tuple(tuple(int(v) for v in row) for row in case["halves"])
        case["votes"] = model.votes(case["p_home"], case["p_dead"], case["out"])
        case["cost"] = cost
        case["tags"] = tags.imitate(matches[file_name.split("_")[0]], zlib.crc32(file_name.encode()))
        CASES.append(case)

    def everyone(weight_dead, lookback, weight_team):
        return [(i, weight_dead, lookback, weight_team) for i in range(len(CASES))]

    print(HEADER)
    show("No tags", run(everyone(0.0, 2.0, 0.0)))
    best, best_row = (0.0, 2.0, 0.0), run(everyone(0.0, 2.0, 0.0))
    for lookback in LOOKBACKS:
        for weight in WEIGHTS_DEAD[1:]:
            row = run(everyone(weight, lookback, 0.0))
            show("dead {:g}, look back {:g} s".format(weight, lookback), row)
            if row[2] > best_row[2]:
                best, best_row = (weight, lookback, 0.0), row
    for weight in WEIGHTS_TEAM[1:]:
        row = run(everyone(best[0], best[1], weight))
        show("best dead, team {:g}".format(weight), row)
        if row[2] > best_row[2]:
            best, best_row = (best[0], best[1], weight), row
    print("\nBEST dead {:g}, look back {:g} s, team {:g}".format(*best))
    show("Best", best_row)


def test(name):
    bundle = model.load_bundle(config.candidate_path(name))
    cost = decode.transition_costs(config.V2_P_TEAM_SWITCH, config.V2_P_TO_DEAD,
                                   config.V2_P_FROM_DEAD)
    versions = bench.versions(external.TEST_MATCHES, bench.HIDDEN_SCENARIOS)
    prepared = []
    for match_id, scenario, version, truth, touchless in versions:
        facts = features.compute(version.tracking)
        p_home, p_dead = bundle.probabilities(version.tracking, facts)
        prepared.append((match_id, scenario, version, truth, touchless,
                         model.votes(p_home, p_dead, facts.out)))

    print(HEADER)
    rows = [measure(decode.decode(v[5], v[2].tracking.halves, cost), v[3], v[4]) for v in prepared]
    show("Held-out, no tags", np.mean(rows, axis=0))
    for label, false_share, timing in QUALITY:
        rows = []
        for match_id, scenario, version, truth, touchless, votes in prepared:
            seed = zlib.crc32((match_id + scenario).encode())
            fake_tags = tags.imitate(version, seed, false_share, timing)
            evidence = tags.add_votes(votes.copy(), fake_tags, config.TAG_WEIGHT_DEAD,
                                      config.TAG_LOOKBACK_S, config.TAG_WEIGHT_TEAM)
            rows.append(measure(decode.decode(evidence, version.tracking.halves, cost), truth,
                                touchless))
        show("Held-out, tags: " + label, np.mean(rows, axis=0))


if __name__ == "__main__":
    candidate = sys.argv[1] if len(sys.argv) > 1 else "v3"
    if "--test" in sys.argv:
        test(candidate)
    else:
        tune(candidate)
