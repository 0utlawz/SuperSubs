"""Compare both versions on rough tracking with real answers: the held-out DFL matches, degraded.

    python tools/rough_check.py

The Clasicos combine rough tracking with a weak answer key, so a result there has two possible
causes. This keeps the official answers and makes only the tracking rough. No setting changes.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
from possession import config, external  # noqa: E402

# (name, hide_ball settings); gaps are run at 3, 12 and 48 s for each.
LEVELS = (
    ("normal", dict(noise_m=0.3)),
    ("rough", dict(noise_m=0.6, camera_half_m=30.0, id_swap_s=30.0)),
    ("very rough", dict(noise_m=0.8, camera_half_m=25.0, id_swap_s=20.0, player_drop=0.15)),
    ("air lost", dict(noise_m=0.6, camera_half_m=30.0, id_swap_s=30.0, air_share=0.8)),
)


def main():
    bundle = bench.load_bundle()
    candidates = bench.load_candidates()
    saved = []
    print("{:<12}{:<11}{:>9}{:>12}{:>9}{:>9}{:>7}".format(
        "tracking", "method", "spellF1", "no-touches", "macroF1", "possess", "dead%"))
    for name, settings in LEVELS:
        results = {"baseline": [], "version1": [], "version2": []}
        for candidate_name, _ in candidates:
            results[candidate_name] = []
        for offset, match_id in enumerate(external.TEST_MATCHES):
            match = external.carry_dead_ball(external.load_match(match_id))
            touchless = bench.without_touches(match.labels, match.tracking.halves)
            for gap_index, gap in enumerate((3.0, 12.0, 48.0)):
                version = external.hide_ball(match, 3000 + offset * 10 + gap_index,
                                             stoppage_bias=4.0, mean_gap_s=gap, **settings)
                tracking = version.tracking
                methods = [("baseline", bench.baseline_labels(tracking)),
                           ("version1", bench.pipeline_labels(tracking)),
                           ("version2", bench.pipeline_v2_labels(tracking, bundle))]
                for candidate_name, candidate in candidates:
                    methods.append((candidate_name, bench.pipeline_v2_labels(tracking, candidate)))
                for method, labels in methods:
                    results[method].append((bench.score(labels, match.labels),
                                            bench.score(labels, touchless)))
        for method, rows in results.items():
            raw = [r[0] for r in rows]
            saved.append({"level": name, "method": method,
                          "spell": float(np.mean([r["spell"] for r in raw])),
                          "macro": float(np.mean([r["macro"] for r in raw])),
                          "possessions": float(np.mean([r["possessions"] for r in raw])),
                          "dead": float(np.mean([r["dead"] for r in raw]))})
            print("{:<12}{:<11}{:>9.3f}{:>12.3f}{:>9.3f}{:>9.0f}{:>6.1f}%".format(
                name, method, np.mean([r["spell"] for r in raw]),
                np.mean([r[1]["spell"] for r in rows]), np.mean([r["macro"] for r in raw]),
                np.mean([r["possessions"] for r in raw]), 100 * np.mean([r["dead"] for r in raw])))

    with open(os.path.join(config.ROOT, "reports", "rough_check.json"), "w") as handle:
        json.dump(saved, handle, indent=1)


if __name__ == "__main__":
    main()
