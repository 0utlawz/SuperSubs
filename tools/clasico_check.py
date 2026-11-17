"""Test the frozen program on two Clasicos with real automatic tracking. No setting changes here.

    python tools/download_skillcorner.py ../external-data/skillcorner
    python tools/clasico_check.py

SkillCorner's first open-data release (MIT; credit SkillCorner): broadcast tracking at 10 frames a
second, with a team-in-possession flag. It is resampled to 25 frames a second so the program sees
the shape it expects. Frames without a flag are not scored, because the flag does not say whether
they are dead balls or simply off camera, so this measures who has the ball and not stoppages.

The flag also runs on through many stoppages, so a correct dead-ball call can count against us.
"team only" removes that: dead-ball calls become "no answer" for every method, and the official
scorer carries the last team across them, so it compares only who the program says has the ball.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import bench  # noqa: E402
from possession import config, data, features  # noqa: E402

SKILLCORNER_DIR = os.environ.get("MTAH_SKILLCORNER_DIR") or os.path.join(
    os.path.dirname(ROOT), "external-data", "skillcorner")
MATCHES = (2841, 3442)
SOURCE_FPS = 10
HALF_GAP = 250


def resample(values, n_frames):
    """Linear resample along time to n_frames; a frame next to a missing value stays missing."""
    position = np.arange(n_frames) * SOURCE_FPS / config.FPS
    low = np.minimum(position.astype(int), len(values) - 2)
    weight = (position - low).reshape((-1,) + (1,) * (values.ndim - 1))
    return (1 - weight) * values[low] + weight * values[low + 1]


def load(match_id):
    """(Tracking, labels, names) for one Clasico at 25 frames a second."""
    folder = os.path.join(SKILLCORNER_DIR, str(match_id))
    with open(os.path.join(folder, "match_data.json")) as handle:
        meta = json.load(handle)
    with open(os.path.join(folder, "structured_data.json")) as handle:
        frames = json.load(handle)

    home_id = meta["home_team"]["id"]
    people = sorted(p["trackable_object"] for p in meta["players"])
    slot_of = {person: slot for slot, person in enumerate(people)}
    side_of = {p["trackable_object"]: data.HOME if p["team_id"] == home_id else data.AWAY
               for p in meta["players"]}
    ball_id = meta["ball"]["trackable_object"]
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])

    blocks = []
    for period in (1, 2):
        kept = [f for f in frames if f.get("period") == period]
        positions = np.full((len(kept), len(people), 2), np.nan)
        ball = np.full((len(kept), 2), np.nan)
        flag = np.zeros(len(kept), dtype=np.int8)  # 0 no flag, 1 home, 2 away
        for row, frame in enumerate(kept):
            group = (frame.get("possession") or {}).get("group")
            flag[row] = {"home team": 1, "away team": 2}.get(group, 0)
            for item in frame.get("data") or []:
                who = item.get("trackable_object")
                if who == ball_id:
                    ball[row] = item["x"], item["y"]
                elif who in slot_of:
                    positions[row, slot_of[who]] = item["x"], item["y"]
        n_frames = int((len(kept) - 1) * config.FPS / SOURCE_FPS)
        nearest = np.round(np.arange(n_frames) * SOURCE_FPS / config.FPS).astype(int)
        blocks.append((resample(positions, n_frames), resample(ball, n_frames), flag[nearest]))

    total = sum(len(b[1]) for b in blocks) + HALF_GAP
    players = np.full((total, len(people), 3), np.nan)
    balls = np.full((total, 3), np.nan)
    labels = np.full(total, "X", dtype="<U1")
    halves, start = [], 0
    scale = np.array([length, width])
    for number, (positions, ball, flag) in enumerate(blocks, 1):
        end = start + len(ball)
        players[start:end, :, :2] = (positions + scale / 2) / scale
        balls[start:end, :2] = (ball + scale / 2) / scale
        labels[start:end] = np.array(["X", "H", "A"])[flag]
        halves.append((number, start, end - 1))
        start = end + HALF_GAP

    velocity = features.central_velocity(players[:, :, :2] * scale, config.VELOCITY_HALF_WINDOW)
    players[:, :, 2] = np.hypot(velocity[:, :, 0], velocity[:, :, 1])
    names = (meta["home_team"]["short_name"], meta["away_team"]["short_name"])
    tracking = data.Tracking(players, balls, np.array([side_of[p] for p in people], dtype=np.int8),
                             names, (length, width), tuple(halves))
    return tracking, labels, names


def agreement(labels, truth):
    """Of the flagged frames, how many we give to a team, and how often that team is right."""
    flagged = truth != "X"
    called = flagged & (labels != "D")
    return float(called.sum() / flagged.sum()), float(np.mean(labels[called] == truth[called]))


def main():
    saved = []
    print("{:<30}{:<10}{:>9}{:>11}{:>10}{:>11}{:>10}".format(
        "match", "method", "spellF1", "team only", "possess", "team call", "team ok"))
    for match_id in MATCHES:
        tracking, truth, names = load(match_id)
        title = "{} vs {}".format(*names)
        print("{:<30}{:<10}{:>9}{:>11}{:>10.0f}".format(
            title, "flag", "-", "-", bench.score(truth, truth)["possessions"]))
        methods = [("baseline", bench.baseline_labels(tracking)),
                   ("version1", bench.pipeline_labels(tracking))]
        bundle = bench.load_bundle()
        if bundle is not None:
            methods.append(("version2", bench.pipeline_v2_labels(tracking, bundle)))
        for candidate_name, candidate in bench.load_candidates():
            methods.append((candidate_name, bench.pipeline_v2_labels(tracking, candidate)))
        for method, labels in methods:
            result = bench.score(labels, truth)
            team_only = bench.score(np.where(labels == "D", "X", labels), truth)
            share, right = agreement(labels, truth)
            saved.append({"match": title, "method": method, "spell": result["spell"],
                          "team_only": team_only["spell"], "possessions": result["possessions"],
                          "flag_possessions": bench.score(truth, truth)["possessions"],
                          "team_call": share, "team_ok": right})
            print("{:<30}{:<10}{:>9.3f}{:>11.3f}{:>10.0f}{:>10.1f}%{:>9.1f}%".format(
                "", method, result["spell"], team_only["spell"], result["possessions"],
                100 * share, 100 * right))

    with open(os.path.join(config.ROOT, "reports", "clasico_check.json"), "w") as handle:
        json.dump(saved, handle, indent=1)


if __name__ == "__main__":
    main()
