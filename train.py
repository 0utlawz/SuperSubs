"""Train version 2 on the five DFL tuning matches, leaving each out in turn.

    python train.py --all                # the adopted model: version 3, three models averaged,
                                         # trained on all seven DFL matches
    python train.py --candidate v9       # a candidate on the five tuning matches, beside it

For each match, two models are trained on the other four and used to predict the one left out, so
the decoder is tuned on predictions from a match the models never saw. A last pair is then trained
on all five. The held-out matches, the Clasicos and the competition match are never read here.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
from possession import config, external, features, model  # noqa: E402

STRIDE = 5  # one frame in five: neighbouring frames say nearly the same thing
MEMBER_SEED_STEP = 100  # averaged members hide the ball and draw noise from different seeds
LIVE_MEMBERS = 3  # the adopted model averages three; fewer fakes on DFL and Metrica, same found
# (touch detection, closing on the ball, contact chain with landing and turnaround, the team's
# expert rules, where players gather, speed worked out from positions) per candidate.
CANDIDATES = {"v3": (True, False, False, False, False, False),
              "v4": (True, True, False, False, False, False),
              "v5": (True, False, True, False, False, False),
              "v6": (True, False, True, True, False, False),
              "v7": (True, False, True, True, True, False),
              "v8": (True, False, True, False, False, True),
              "v9": (True, False, True, False, False, True),
              "v10": (True, False, True, False, False, True),
              "v11": (True, False, True, False, False, True),
              "v12": (True, False, True, False, False, True),
              "v13": (True, False, True, False, False, True)}

# Candidates that also train the change detector.
WITH_CHANGE = ("v12",)

# Which matches a candidate learns from, when not the five DFL tuning matches.
MATCH_SETS = {"v11": external.TRAIN_MATCHES + external.TEST_MATCHES + external.METRICA_MATCHES,
              "v12": external.TRAIN_MATCHES + external.TEST_MATCHES + external.METRICA_MATCHES,
              "v13": external.TRAIN_MATCHES + external.TEST_MATCHES + external.PFF_MATCHES}

# The training target removes spells shorter than this. The team's rule is 15 frames of control.
TARGET_TOUCH_S = {"v6": 15.0 / config.FPS, "v7": 15.0 / config.FPS,
                  "v10": 2.0}  # the team's rule of 11 November: 2 s on the ball for a switch

# (seed, hide_ball settings). Training sees a spread of conditions, from close to the official data
# to as rough as automatic tracking gets, and never the same random hiding as the cases it is
# scored on. Automatic tracking loses players and swaps tracks, so most variants do too.
TRAIN_VARIANTS = (
    (1001, dict(stoppage_bias=1.0, mean_gap_s=3.0, noise_m=0.3)),
    (1002, dict(stoppage_bias=4.0, mean_gap_s=12.0, noise_m=0.3, id_swap_s=60.0, air_share=0.5)),
    (1003, dict(stoppage_bias=1.0, mean_gap_s=48.0, noise_m=0.8, camera_half_m=30.0, id_swap_s=60.0)),
    (1004, dict(stoppage_bias=4.0, mean_gap_s=48.0, noise_m=0.5, camera_half_m=30.0, id_swap_s=30.0,
                air_share=0.5)),
    (1005, dict(stoppage_bias=4.0, mean_gap_s=12.0, noise_m=0.6, camera_half_m=25.0, id_swap_s=30.0,
                air_share=0.8)),
)

# Rough cases the decoder is also tuned on, one per gap length.
ROUGH_SCENARIOS = (("rough_3s", 3.0), ("rough_12s", 12.0), ("rough_48s", 48.0))


def rows_for(match_id, options, touch_s, change=False, member=0):
    """(team_x, team_y, dead_x, dead_y) for one match, across every training variant."""
    match = external.carry_dead_ball(external.load_match(match_id))
    halves = match.tracking.halves
    truth = bench.without_touches(match.labels, halves, touch_s)
    keep = np.zeros(len(truth), dtype=bool)
    natural = external.is_natural(match_id)
    # Broadcast tracking is used once, as it comes, so it takes every frame to weigh as much as a
    # DFL match seen five ways at one frame in five.
    keep[(0 if natural else member % STRIDE)::(1 if natural else STRIDE)] = True
    alive = keep & np.isin(truth, ("H", "A"))
    graded = keep & (truth != "X")

    team_x, team_y, dead_x, dead_y, change_x, change_y = [], [], [], [], [], []
    changing = model.change_targets(truth, halves)
    for seed, settings in ((None, None),) if natural else TRAIN_VARIANTS:
        version = match if natural else external.hide_ball(
            match, seed + MEMBER_SEED_STEP * member, **settings)
        tracking = version.tracking
        features_home = model.team_features(tracking, features.compute(tracking), *options)[0]
        team_x.append(features_home[alive])
        team_y.append(truth[alive] == "H")
        if change:
            change_x.append(features_home[graded])
            change_y.append(changing[graded])
        # The same frames with the sides swapped, so neither 'home' nor one club is learned.
        swapped = tracking._replace(team_of_slot=1 - tracking.team_of_slot)
        features_away = model.team_features(swapped, features.compute(swapped), *options)[0]
        team_x.append(features_away[alive])
        team_y.append(truth[alive] == "A")
        dead_x.append(model.dead_features(tracking, options[5])[0][graded])
        dead_y.append(truth[graded] == "D")
    rows = [np.concatenate(team_x), np.concatenate(team_y), np.concatenate(dead_x),
            np.concatenate(dead_y)]
    if change:
        rows += [np.concatenate(change_x), np.concatenate(change_y)]
    return tuple(rows)


def fit_on(rows, match_ids, options, seed=0):
    parts = [rows[m] for m in match_ids]
    columns = [np.concatenate([p[i] for p in parts]) for i in range(len(parts[0]))]
    change = columns[4:6] if len(columns) > 4 else (None, None)
    return model.fit(*columns[:4], *options, change_x=change[0], change_y=change[1], seed=seed)


def fit_members(member_rows, match_ids, options):
    """One model, or an average of several when more than one member was prepared."""
    if len(member_rows) == 1:
        return fit_on(member_rows[0], match_ids, options)
    return model.Ensemble([fit_on(rows, match_ids, options, seed=member)
                           for member, rows in enumerate(member_rows)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", help="save as models/possession_<name>.pkl instead")
    parser.add_argument("--all", action="store_true",
                        help="the adopted model: version 3, trained on all seven DFL matches")
    parser.add_argument("--members", type=int,
                        help="average this many models, each trained on differently degraded data "
                             "(default {} with --all, else 1)".format(LIVE_MEMBERS))
    args = parser.parse_args()
    if args.members is None:
        args.members = LIVE_MEMBERS if args.all else 1
    name = "v9" if args.all else args.candidate
    options = CANDIDATES.get(name, (False,) * 6)
    touch_s = TARGET_TOUCH_S.get(name, bench.TOUCH_S)
    matches = (external.TRAIN_MATCHES + external.TEST_MATCHES if args.all
               else MATCH_SETS.get(name, external.TRAIN_MATCHES))
    suffix = "_live" if args.all else ("_" + args.candidate if args.candidate else "")
    if args.members != (LIVE_MEMBERS if args.all else 1):
        suffix += "_ens{}".format(args.members)
    oof_dir = os.path.join(config.ROOT, "reports", "oof" + suffix)
    model_path = (config.MODEL_PATH if suffix == "_live" or not suffix
                  else config.candidate_path(suffix.strip("_")))

    started = time.time()
    with_change = name in WITH_CHANGE
    member_rows = [{m: rows_for(m, options, touch_s, with_change, member) for m in matches}
                   for member in range(args.members)]
    print("rows ready: {} member(s), {:,} team rows each  ({:.0f} s)".format(
        args.members, sum(len(r[0]) for r in member_rows[0].values()), time.time() - started),
        flush=True)

    os.makedirs(oof_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    for left_out in matches:
        others = [m for m in matches if m != left_out]
        bundle = fit_members(member_rows, others, options)
        cases = bench.versions((left_out,), bench.HIDDEN_SCENARIOS)
        match = external.carry_dead_ball(external.load_match(left_out))
        for offset, (name, gap) in enumerate(() if external.is_natural(left_out) else ROUGH_SCENARIOS):
            rough = external.hide_ball(match, 2000 + offset, stoppage_bias=4.0, mean_gap_s=gap,
                                       noise_m=0.6, camera_half_m=30.0, id_swap_s=30.0,
                                       air_share=0.5)
            cases.append((left_out, name, rough, cases[0][3], cases[0][4]))
        for match_id, name, version, truth, touchless in cases:
            facts = features.compute(version.tracking)
            p_home, p_dead = bundle.probabilities(version.tracking, facts)
            p_change = bundle.change_probability(version.tracking, facts)
            extra = {} if p_change is None else {"p_change": p_change.astype(np.float32)}
            np.savez(os.path.join(oof_dir, "{}_{}.npz".format(match_id, name)),
                     p_home=p_home.astype(np.float32), p_dead=p_dead.astype(np.float32),
                     out=facts.out, cue=features.restart_cue(facts, version.tracking.pitch),
                     truth=truth, touchless=touchless, **extra,
                     halves=np.array(version.tracking.halves))
        print("left out {} done ({:.0f} s)".format(left_out, time.time() - started), flush=True)

    bundle = fit_members(member_rows, matches, options)
    with open(model_path, "wb") as handle:
        pickle.dump(bundle, handle)
    print("final model saved ({:.0f} s)".format(time.time() - started), flush=True)


if __name__ == "__main__":
    main()
