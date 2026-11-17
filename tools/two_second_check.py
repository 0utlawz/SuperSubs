"""Does the team's 2-second rule for a change of possession gain or lose points?

    python tools/two_second_check.py

Four ways of applying it, each scored on the held-out DFL matches against three versions of the
official answers: every possession, possessions of 1 s or more, and possessions of 2 s or more.
No setting changes here.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
from possession import checker, config, external, model  # noqa: E402

MIN_POSSESSION_S = 2.0
TRUTHS = (("every possession", 0.0), ("1 s or more", 1.0), ("2 s or more", 2.0))


def at_least(labels, halves, seconds):
    """Reported possessions shorter than this are handed back to their neighbours."""
    out = labels.copy()
    for _, first, last in halves:
        parts = checker.segments(out, first, last)
        for index, (start, stop, label) in enumerate(parts):
            if label not in ("H", "A") or stop - start >= seconds * config.FPS:
                continue
            before = parts[index - 1][2] if index > 0 else None
            after = parts[index + 1][2] if index + 1 < len(parts) else None
            replacement = before if before in ("H", "A") else after
            if replacement is None:
                replacement = "D"
            out[start:stop] = replacement
    return out


def main():
    bundles = {name: model.load_bundle(config.candidate_path(name))
               for name in ("v9", "v10")}
    methods = (("Version 3 as it is", "v9", False), ("Train on 2 s", "v10", False),
               ("Apply 2 s afterwards", "v9", True), ("Both", "v10", True))
    results = {}
    for match_id, scenario, version, truth, _ in bench.versions(external.TEST_MATCHES,
                                                                bench.HIDDEN_SCENARIOS):
        halves = version.tracking.halves
        truths = [truth if s == 0 else bench.without_touches(truth, halves, s) for _, s in TRUTHS]
        labels = {name: model.label_match(version.tracking, bundle) for name, bundle in bundles.items()}
        for title, name, after in methods:
            chosen = at_least(labels[name], halves, MIN_POSSESSION_S) if after else labels[name]
            results.setdefault(title, []).append(
                [bench.score(chosen, t)["spell"] for t in truths]
                + [bench.score(chosen, truth)["possessions"]])
        for (key, seconds), t in zip(TRUTHS, truths):
            results.setdefault("true count, " + key, []).append(bench.score(t, t)["possessions"])

    print("Held-out DFL matches, mean of six hidden-ball conditions. spellF1 against:")
    print("{:<24}{:>18}{:>14}{:>14}{:>11}".format("", *[k for k, _ in TRUTHS], "reported"))
    for title, _, _ in methods:
        row = np.mean(results[title], axis=0)
        print("{:<24}{:>18.3f}{:>14.3f}{:>14.3f}{:>11.0f}".format(title, *row))
    print("true possessions per match: " + ", ".join(
        "{} {:.0f}".format(key, np.mean(results["true count, " + key])) for key, _ in TRUTHS))


if __name__ == "__main__":
    main()
