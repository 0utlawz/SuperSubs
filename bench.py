"""Score the pipeline on external matches with the official metric code.

    python bench.py               # the five tuning matches
    python bench.py --test        # the two matches held out of all tuning

Every number here comes from the DFL open dataset, never from the competition match. Two truths
are scored: DFL's own possession flag, and the same flag with brief touches removed, because the
competition says that a touch without control is not a change of possession.
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from possession import config, decode, evidence, external, features, model  # noqa: E402

# A spell of the other team shorter than this, with the same team on both sides of it, counts as a
# touch without control rather than a change of possession (CHALLENGE.md, "What does not count").
TOUCH_S = 1.0

# (name, stoppage_bias, mean gap in seconds); a bias of None leaves the DFL tracking as it is. Gap
# length is not published for the competition data, so a wide spread is covered instead of one value.
SCENARIOS = (("clean", None, None),
             ("gaps_3s", 1.0, 3.0), ("gaps_12s", 1.0, 12.0), ("gaps_48s", 1.0, 48.0),
             ("gaps_3s_stops", 4.0, 3.0), ("gaps_12s_stops", 4.0, 12.0),
             ("gaps_48s_stops", 4.0, 48.0))
HIDDEN_SCENARIOS = SCENARIOS[1:]


def _official_metrics():
    path = os.path.join(config.OFFICIAL_REPO, "evaluation", "possession_metrics.py")
    spec = importlib.util.spec_from_file_location("possession_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PM = _official_metrics()


def without_touches(labels, halves, touch_s=TOUCH_S):
    """The truth with brief spells of the other team merged into the possession around them."""
    out = labels.copy()
    limit = touch_s * config.FPS
    for _, first, last in halves:
        block = out[first:last + 1]
        starts = np.flatnonzero(np.concatenate(([True], block[1:] != block[:-1])))
        ends = np.concatenate((starts[1:], [len(block)]))
        for i in range(1, len(starts) - 1):
            before, here, after = block[starts[i - 1]], block[starts[i]], block[starts[i + 1]]
            if ends[i] - starts[i] < limit and before == after and {before, here} == {"H", "A"}:
                block[starts[i]:ends[i]] = before
    return out


def score(labels, truth):
    """The official spellF1 and macroF1, plus the shape of what was written."""
    gt = PM.label_codes(truth)
    pred = PM.label_codes(labels)
    runs = PM.valid_runs(gt)
    filled = PM.fill_abstentions(pred, runs)
    spells = PM.spell_scores(gt, filled, runs)
    graded = truth != "X"
    block_starts = np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1])))
    return {
        "spell": spells["mean_f1"],
        "macro": PM.macro_f1(PM.scored_confusion(PM.confusion(gt, pred))),
        "possessions": spells["n_pred"],
        "stoppages": int(np.sum(labels[block_starts] == "D")),
        "dead": float(np.mean(labels[graded] == "D")),
    }


def baseline_labels(tracking, window=50):
    """The organisers' floor: nearest player to the ball, majority vote over +/- window frames."""
    length, width = tracking.pitch
    scale = np.array([length, width])
    pos = tracking.players[:, :, :2] * scale
    ball = tracking.ball[:, :2] * scale
    seen = np.isfinite(ball[:, 0])
    dist = np.hypot(pos[:, :, 0] - ball[:, None, 0], pos[:, :, 1] - ball[:, None, 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    team = tracking.team_of_slot[dist.argmin(1)]

    home = np.concatenate(([0], np.cumsum(seen & (team == 0))))
    away = np.concatenate(([0], np.cumsum(seen & (team == 1))))
    index = np.arange(len(ball))
    low = np.maximum(0, index - window)
    high = np.minimum(len(ball), index + window + 1)
    h, a = home[high] - home[low], away[high] - away[low]
    labels = np.where(h + a == 0, "D", np.where(h >= a, "H", "A"))
    return np.where(features.graded_mask(len(ball), tracking.halves), labels, "X")


def pipeline_labels(tracking):
    votes = evidence.build(features.compute(tracking), [])  # no Smart Tagging on external matches
    return decode.decode(votes, tracking.halves)


def pipeline_v2_labels(tracking, bundle):
    return model.label_match(tracking, bundle, restart_cue=False)


def load_bundle():
    return model.load_bundle() if os.path.isfile(config.MODEL_PATH) else None


def load_candidates():
    """[(name, bundle)] for the models named in MTAH_CANDIDATE, to compare beside the adopted one."""
    names = [n for n in os.environ.get("MTAH_CANDIDATE", "").split(",") if n]
    return [(n, model.load_bundle(config.candidate_path(n))) for n in names]


def versions(match_ids, scenarios=SCENARIOS, seed=0):
    """[(match, scenario name, degraded Match, truth, truth without touches)]."""
    out = []
    for offset, match_id in enumerate(match_ids):
        match = external.carry_dead_ball(external.load_match(match_id))
        clean_truth = without_touches(match.labels, match.tracking.halves)
        if external.is_natural(match_id):
            out.append((match_id, "natural", match, match.labels, clean_truth))
            continue
        for name, bias, gap in scenarios:
            version = match if bias is None else external.hide_ball(
                match, seed + offset, stoppage_bias=bias, mean_gap_s=gap)
            out.append((match_id, name, version, match.labels, clean_truth))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", action="store_true", help="score the held-out matches")
    args = parser.parse_args()
    match_ids = external.TEST_MATCHES if args.test else external.TRAIN_MATCHES

    rows = {}
    bundle = load_bundle()
    candidates = load_candidates()
    for _, name, version, truth, touchless in versions(match_ids):
        methods = [("baseline", baseline_labels(version.tracking)),
                   ("pipeline", pipeline_labels(version.tracking))]
        if bundle is not None:
            methods.append(("live", pipeline_v2_labels(version.tracking, bundle)))
        for candidate_name, candidate in candidates:
            methods.append((candidate_name, pipeline_v2_labels(version.tracking, candidate)))
        for method, labels in methods:
            raw, merged = score(labels, truth), score(labels, touchless)
            rows.setdefault((name, method), []).append((raw, merged))

    table = []
    true_stats = [score(truth, truth) for truth in
                  (external.load_match(m).labels for m in match_ids)]
    print("{} matches. Truth: {:.0f} possessions, {:.0f} stoppages, {:.1f}% dead ball per match".format(
        "held-out" if args.test else "tuning", np.mean([s["possessions"] for s in true_stats]),
        np.mean([s["stoppages"] for s in true_stats]),
        100 * np.mean([s["dead"] for s in true_stats])))
    print("\n{:<18}{:<10}{:>9}{:>12}{:>9}{:>9}{:>11}{:>7}".format(
        "condition", "method", "spellF1", "no-touches", "macroF1", "possess", "stoppages", "dead%"))
    for (name, method), results in rows.items():
        raw = [r[0] for r in results]
        table.append({"condition": name, "method": method,
                      "spell": float(np.mean([r["spell"] for r in raw])),
                      "spell_no_touches": float(np.mean([r[1]["spell"] for r in results])),
                      "macro": float(np.mean([r["macro"] for r in raw])),
                      "possessions": float(np.mean([r["possessions"] for r in raw])),
                      "stoppages": float(np.mean([r["stoppages"] for r in raw])),
                      "dead": float(np.mean([r["dead"] for r in raw]))})
        print("{:<18}{:<10}{:>9.3f}{:>12.3f}{:>9.3f}{:>9.0f}{:>11.0f}{:>6.1f}%".format(
            name, method, np.mean([r["spell"] for r in raw]),
            np.mean([r[1]["spell"] for r in results]), np.mean([r["macro"] for r in raw]),
            np.mean([r["possessions"] for r in raw]), np.mean([r["stoppages"] for r in raw]),
            100 * np.mean([r["dead"] for r in raw])))

    os.makedirs(os.path.join(config.ROOT, "reports"), exist_ok=True)
    with open(os.path.join(config.ROOT, "reports", "bench_{}.json".format(
            "test" if args.test else "tuning")), "w") as handle:
        json.dump({"truth": {key: float(np.mean([s[key] for s in true_stats]))
                             for key in ("possessions", "stoppages", "dead")}, "rows": table},
                  handle, indent=1)


if __name__ == "__main__":
    main()
