"""How accurate is the live program? Every DFL match left out once, plus the two Clasicos.

    python train.py --all && python tools/accuracy_report.py
    python tools/accuracy_report.py --oof oof_v11            # another candidate's left-out results
    python tools/accuracy_report.py --model possession.pkl --matches MET1,MET2,MET3

DFL: each match is scored with models trained on the other six, so no match judges itself. The
votes are exactly the live ones, with imitation Smart Tagging and the team's restart rules.
Clasicos: the final model, which never saw them, against SkillCorner's automatic flag. The last table
puts the key numbers side by side so the team can decide where the sources disagree. Nothing here
reads the competition match or its answer key.
"""

import argparse
import glob
import json
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import clasico_check  # noqa: E402
from possession import config, decode, external, model, tags  # noqa: E402

PM = bench.PM
OOF_DIR = os.path.join(config.ROOT, "reports", "oof_live")
MATCHES = None  # set from --matches: which left-out matches to include
FULL_MATCH_FRAMES = 90 * 60 * config.FPS


def spells(labels, truth):
    """(reported, true, found) possessions, matched at IoU 0.5."""
    gt = PM.label_codes(truth)
    pred = PM.label_codes(labels)
    runs = PM.valid_runs(gt)
    filled = PM.fill_abstentions(pred, runs)
    gt_spells = PM.possession_spells(gt, runs)
    pred_spells = PM.possession_spells(filled, runs)
    found = PM.matched_gt_flags(PM.spell_overlaps(gt_spells, pred_spells), len(gt_spells), 0.5)
    return len(pred_spells), len(gt_spells), int(np.sum(found))


def shape(labels, graded):
    """What a set of labels says about the match, per 90 minutes of play."""
    per_90 = FULL_MATCH_FRAMES / graded.sum()
    lengths, stoppages = [], 0
    starts = np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1]))) if len(labels) else []
    ends = np.concatenate((starts[1:], [len(labels)]))
    for start, end in zip(starts, ends):
        if not graded[start]:
            continue
        if labels[start] in ("H", "A"):
            lengths.append((end - start) / config.FPS)
        elif labels[start] == "D":
            stoppages += 1
    lengths = np.array(lengths) if lengths else np.array([0.0])
    return {"possessions per 90": len(lengths) * per_90, "median length s": float(np.median(lengths)),
            "mean length s": float(np.mean(lengths)), "under 2 s": float(np.mean(lengths < 2)),
            "stoppages per 90": stoppages * per_90,
            "dead share": float(np.mean(labels[graded] == "D"))}


def scored(case, name, match):
    """The per-case figures, from left-out probabilities with the live votes on top."""
    halves = tuple(tuple(int(v) for v in row) for row in case["halves"])
    evidence = model.votes(case["p_home"], case["p_dead"], case["out"], cue=case["cue"])
    evidence = tags.add_votes(evidence, tags.imitate(match, zlib.crc32(name.encode())),
                              config.TAG_WEIGHT_DEAD, config.TAG_LOOKBACK_S, config.TAG_WEIGHT_TEAM)
    cost = decode.transition_costs(config.V2_P_TEAM_SWITCH, config.V2_P_TO_DEAD,
                                   config.V2_P_FROM_DEAD)
    return decode.decode(evidence, halves, cost)


def predicted_cases(bundle, match_ids):
    """Left-out style cases for a fixed model on matches it never saw: nine ways of hiding the ball."""
    import train
    from possession import features
    cases = []
    for match_id in match_ids:
        match = external.carry_dead_ball(external.load_match(match_id))
        versions = [(scenario, version) for _, scenario, version, _, _ in
                    bench.versions((match_id,), bench.HIDDEN_SCENARIOS)]
        rough = () if external.is_natural(match_id) else train.ROUGH_SCENARIOS
        for offset, (scenario, gap) in enumerate(rough):
            versions.append((scenario, external.hide_ball(
                match, 2000 + offset, stoppage_bias=4.0, mean_gap_s=gap, noise_m=0.6,
                camera_half_m=30.0, id_swap_s=30.0, air_share=0.5)))
        for scenario, version in versions:
            facts = features.compute(version.tracking)
            p_home, p_dead = bundle.probabilities(version.tracking, facts)
            cases.append(("{}_{}.npz".format(match_id, scenario), {
                "p_home": p_home, "p_dead": p_dead, "out": facts.out,
                "cue": features.restart_cue(facts, version.tracking.pitch),
                "truth": match.labels, "halves": np.array(version.tracking.halves)}))
    return cases


def dfl_rows(cases=None):
    matches = {}
    per_match = {}
    shapes = {"truth": [], "ours": []}
    if cases is None:
        cases = [(os.path.basename(path), dict(np.load(path)))
                 for path in sorted(glob.glob(os.path.join(OOF_DIR, "*.npz")))]
    for name, case in cases:
        match_id = name.split("_")[0]
        if MATCHES and match_id not in MATCHES:
            continue
        if match_id not in matches:
            matches[match_id] = external.load_match(match_id)
        labels = scored(case, name, matches[match_id])
        truth = case["truth"]
        graded = truth != "X"
        alive_both = graded & np.isin(truth, ("H", "A")) & np.isin(labels, ("H", "A"))
        reported, true, found = spells(labels, truth)
        score = bench.score(labels, truth)
        per_match.setdefault(match_id, []).append({
            "frames right": float(np.mean(labels[graded] == truth[graded])),
            "team right": float(np.mean(labels[alive_both] == truth[alive_both])),
            "dead caught": float(np.mean(labels[graded & (truth == "D")] == "D")),
            "spellF1": score["spell"], "macroF1": score["macro"],
            "found": found / true, "reported": reported, "true": true,
            "fake": reported - found, "dead share": score["dead"],
            "true dead share": float(np.mean(truth[graded] == "D"))})
        shapes["truth"].append(shape(truth, graded))
        shapes["ours"].append(shape(labels, graded))
    return per_match, shapes


def clasico_rows():
    bundle = model.load_bundle()
    rows, shapes = [], {"flag": [], "ours": []}
    for match_id in clasico_check.MATCHES:
        tracking, flag, names = clasico_check.load(match_id)
        labels = model.label_match(tracking, bundle)
        flagged = flag != "X"
        called = flagged & np.isin(labels, ("H", "A"))
        graded = np.zeros(len(flag), dtype=bool)
        for _, first, last in tracking.halves:
            graded[first:last + 1] = True
        team_only = bench.score(np.where(labels == "D", "X", labels), flag)["spell"]
        rows.append({"match": "{} vs {}".format(*names), "spellF1": bench.score(labels, flag)["spell"],
                     "team only": team_only, "team right": float(np.mean(labels[called] == flag[called])),
                     "dead on flagged frames": float(np.mean(labels[flagged] == "D")),
                     "reported": bench.score(labels, flag)["possessions"],
                     "flag": bench.score(flag, flag)["possessions"]})
        flag_with_gaps = np.where(flagged, flag, "D")
        shapes["flag"].append(shape(flag_with_gaps, graded))
        shapes["ours"].append(shape(labels, graded))
    return rows, shapes


def main():
    global OOF_DIR, MATCHES
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oof", help="left-out results folder in reports/ (default oof_live)")
    parser.add_argument("--model", help="score this saved model on --matches instead")
    parser.add_argument("--matches", help="comma-separated match ids to include")
    parser.add_argument("--quick", action="store_true", help="skip the Clasicos and side table")
    args = parser.parse_args()
    if args.oof:
        OOF_DIR = os.path.join(config.ROOT, "reports", args.oof)
    if args.matches:
        MATCHES = args.matches.split(",")
    cases = None
    if args.model:
        cases = predicted_cases(model.load_bundle(os.path.join(config.MODEL_DIR, args.model)),
                                MATCHES)
    per_match, dfl_shapes = dfl_rows(cases)
    print("Each match scored by a model that never saw it "
          "(mean of nine ways of hiding the ball)")
    columns = ("frames right", "team right", "dead caught", "found", "spellF1", "macroF1")
    print("{:<8}".format("match") + "".join("{:>13}".format(c) for c in columns)
          + "{:>11}{:>8}{:>7}".format("reported", "true", "fake"))
    everything = []
    for match_id, cases in per_match.items():
        mean = {k: float(np.mean([c[k] for c in cases])) for k in cases[0]}
        everything.append(mean)
        print("{:<8}".format(match_id) + "".join(
            "{:>12.1f}%".format(100 * mean[c]) if c not in ("spellF1", "macroF1")
            else "{:>13.3f}".format(mean[c]) for c in columns)
              + "{:>11.0f}{:>8.0f}{:>7.0f}".format(mean["reported"], mean["true"], mean["fake"]))
    overall = {k: float(np.mean([m[k] for m in everything])) for k in everything[0]}
    print("{:<8}".format("ALL") + "".join(
        "{:>12.1f}%".format(100 * overall[c]) if c not in ("spellF1", "macroF1")
        else "{:>13.3f}".format(overall[c]) for c in columns)
          + "{:>11.0f}{:>8.0f}{:>7.0f}".format(overall["reported"], overall["true"], overall["fake"]))
    if args.quick:
        return

    clasicos, clasico_shapes = clasico_rows()
    print("\nCLASICOS: final model, never trained on them; answer key = SkillCorner's automatic flag")
    for row in clasicos:
        print("  {:<30} spellF1 {:.3f}  team only {:.3f}  team right {:.1f}%  dead on flagged "
              "frames {:.1f}%  reported {:.0f} (flag {:.0f})".format(
                  row["match"], row["spellF1"], row["team only"], 100 * row["team right"],
                  100 * row["dead on flagged frames"], row["reported"], row["flag"]))

    print("\nKEY NUMBERS SIDE BY SIDE, per 90 minutes of play (the team decides where they disagree)")
    sources = (("Bundesliga truth (DFL)", dfl_shapes["truth"]), ("Our program on Bundesliga", dfl_shapes["ours"]),
               ("Clasico flag (SkillCorner)", clasico_shapes["flag"]),
               ("Our program on Clasicos", clasico_shapes["ours"]))
    keys = ("possessions per 90", "median length s", "mean length s", "under 2 s", "stoppages per 90",
            "dead share")
    print("{:<30}".format("") + "".join("{:>19}".format(k) for k in keys))
    saved = {"dfl_overall": overall, "dfl_per_match": {m: {k: float(np.mean([c[k] for c in v]))
                                                           for k in v[0]} for m, v in per_match.items()},
             "clasicos": clasicos, "sides": {}}
    for name, rows in sources:
        mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        saved["sides"][name] = mean
        print("{:<30}".format(name) + "".join(
            "{:>18.1f}%".format(100 * mean[k]) if k in ("under 2 s", "dead share")
            else "{:>19.1f}".format(mean[k]) for k in keys))
    print("Clasico flag: frames without a flag are counted as stoppages, which over-counts them.")
    with open(os.path.join(config.ROOT, "reports", "accuracy.json"), "w") as handle:
        json.dump(saved, handle, indent=1)


if __name__ == "__main__":
    main()
