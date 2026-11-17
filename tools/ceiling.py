"""Where is the score being lost? Pretend one problem is solved perfectly and see what it is worth.

    python tools/ceiling.py

Each row replaces part of version 1's output with the truth, on the DFL tuning matches only. These
are ceilings, not results: nothing can reach them. They rank what is worth building next, and this
is the test that pointed us at version 2.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
from possession import config, external  # noqa: E402

TEAMS = np.array(["H", "A"])
BALL_MISSING = (0.41, 0.30, 0.20, 0.10, 0.0)


def fill_alive(labels, halves):
    """Give each frame marked "?" the last team named in its half, or the next if there is none."""
    out = labels.copy()
    for _, first, last in halves:
        block = out[first:last + 1]
        known = np.flatnonzero(block != "?")
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(block != "?", np.arange(len(block)), -1))
        carry[carry < 0] = known[0]
        team_only = np.where(np.isin(block, TEAMS), np.arange(len(block)), -1)
        carry = np.maximum.accumulate(team_only)
        first_team = np.flatnonzero(team_only >= 0)
        carry[carry < 0] = first_team[0] if len(first_team) else 0
        missing = block == "?"
        block[missing] = block[carry[missing]]
    return out


def perfect_team(labels, truth, where):
    out = labels.copy()
    change = where & np.isin(out, TEAMS) & np.isin(truth, TEAMS)
    out[change] = truth[change]
    return out


def perfect_dead(labels, truth, where, halves):
    out = labels.copy()
    out[where & (truth == "D")] = "D"
    out[where & (truth != "D") & (out == "D") & (truth != "X")] = "?"
    return fill_alive(out, halves)


def main():
    rows = {}

    def note(name, labels, case):
        for truth in case[3:5]:
            rows.setdefault(name, []).append(bench.score(labels, truth)["spell"])

    for case in bench.versions(external.TRAIN_MATCHES, bench.HIDDEN_SCENARIOS):
        tracking = case[2].tracking
        labels = bench.pipeline_labels(tracking)
        truth = case[3]
        everywhere = truth != "X"
        hidden = everywhere & ~np.isfinite(tracking.ball[:, 0])
        halves = tracking.halves

        note("Today", labels, case)
        note("Perfect team, ball hidden frames only", perfect_team(labels, truth, hidden), case)
        note("Perfect dead ball, ball hidden frames only",
             perfect_dead(labels, truth, hidden, halves), case)
        note("Both of those: perfect when the ball is hidden",
             perfect_team(perfect_dead(labels, truth, hidden, halves), truth, hidden), case)
        note("Perfect team, everywhere", perfect_team(labels, truth, everywhere), case)
        note("Perfect dead ball, everywhere",
             perfect_dead(labels, truth, everywhere, halves), case)

    print("{:<52}{:>9}{:>9}".format("if this were solved perfectly", "spellF1", "gain"))
    base = np.mean(rows["Today"])
    for name, values in rows.items():
        print("{:<52}{:>9.3f}{:>+9.3f}".format(name, np.mean(values), np.mean(values) - base))

    saved = {"ceilings": {name: float(np.mean(values)) for name, values in rows.items()},
             "ball_missing": {}}
    print("\nif more of the ball could be found (real program, nothing perfect):")
    print("{:<30}{:>9}".format("ball missing on", "spellF1"))
    for share in BALL_MISSING:
        scores = []
        for offset, match_id in enumerate(external.TRAIN_MATCHES):
            match = external.carry_dead_ball(external.load_match(match_id))
            touchless = bench.without_touches(match.labels, match.tracking.halves)
            for bias in (1.0, 4.0):
                version = external.hide_ball(match, offset, stoppage_bias=bias, mean_gap_s=12.0,
                                             share=share)
                labels = bench.pipeline_labels(version.tracking)
                scores += [bench.score(labels, t)["spell"] for t in (match.labels, touchless)]
        saved["ball_missing"]["{:.0f}".format(100 * share)] = float(np.mean(scores))
        print("{:<30}{:>9.3f}".format("{:.0f}% of frames".format(100 * share), np.mean(scores)))

    with open(os.path.join(config.ROOT, "reports", "ceiling.json"), "w") as handle:
        json.dump(saved, handle, indent=1)


if __name__ == "__main__":
    main()
