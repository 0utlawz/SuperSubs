"""Which kinds of stoppage do our dead-ball fakes fall in?

    python tools/dead_fake_types.py v3

Uses predictions for DFL matches left out of training. "Dead" here is DFL's official ball-status
flag on each frame. The stoppage type comes from where the ball is when play resumes, so a free kick
after a foul and one after an offside look the same.
"""

import glob
import multiprocessing
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
from possession import config, decode, external, features, model, tags  # noqa: E402

OOF_DIR = os.path.join(config.ROOT, "reports", "oof")
CASES = []


def _run(index):
    path, case = CASES[index]
    match_id = os.path.basename(path).split("_")[0]
    match = external.load_match(match_id)
    length, width = match.tracking.pitch
    ball = match.tracking.ball[:, :2] * np.array(match.tracking.pitch)
    halves = tuple(tuple(int(v) for v in row) for row in case["halves"])
    evidence = model.votes(case["p_home"], case["p_dead"], case["out"])
    cost = decode.transition_costs(config.V2_P_TEAM_SWITCH, config.V2_P_TO_DEAD,
                                   config.V2_P_FROM_DEAD)
    labels = decode.decode(evidence, halves, cost)
    truth = case["truth"]

    counts = {}
    for _, first, last in halves:
        for start, end in zip(*features.runs(truth[first:last + 1] == "D")):
            start, end = start + first, end + first
            if end > last:
                continue
            kind = tags.restart_type(ball[end, 0] - length / 2, ball[end, 1] - width / 2,
                                            length, width)
            entry = counts.setdefault(kind, [0, 0, 0, 0.0])
            entry[0] += 1
            window = labels[start:end]
            called_dead = window == "D"
            entry[3] += float(called_dead.mean())
            alive_starts, alive_ends = features.runs(~called_dead)
            chunks = [(a, b) for a, b in zip(alive_starts, alive_ends) if (b - a) >= config.FPS]
            entry[1] += 1 if chunks else 0
            entry[2] += len(chunks)
    return counts


def main():
    folder = OOF_DIR + ("_" + sys.argv[1] if len(sys.argv) > 1 else "")
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        CASES.append((path, dict(np.load(path))))
    with multiprocessing.get_context("fork").Pool(8) as pool:
        results = pool.map(_run, range(len(CASES)))

    total = {}
    for counts in results:
        for kind, entry in counts.items():
            row = total.setdefault(kind, [0, 0, 0, 0.0])
            for i in range(4):
                row[i] += entry[i]
    n = len(results)
    print("{:<20}{:>11}{:>16}{:>18}{:>16}".format(
        "stoppage ending in", "per match", "called 'in play'", "share of those", "frames caught"))
    print("{:<20}{:>11}{:>16}{:>18}{:>16}".format("", "", "1 s+ per match", "stoppages", "as dead"))
    for kind, row in sorted(total.items(), key=lambda kv: -kv[1][0]):
        print("{:<20}{:>11.1f}{:>16.1f}{:>17.0f}%{:>15.0f}%".format(
            kind, row[0] / n, row[1] / n, 100 * row[1] / row[0], 100 * row[3] / row[0]))


if __name__ == "__main__":
    main()
