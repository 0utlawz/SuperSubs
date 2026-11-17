"""How much proof a change of team should need, under different readings of "control".

    python tools/proof_table.py

The DFL possession flag counts very brief spells, while the competition says a touch without
control is not a change of possession. So the right proof depends on how strictly "control" is
read. This scores each proof against the truth with spells shorter than 0, 1, 2, 3 and 5 seconds
merged away, on the tuning matches only, at normal and at heavy tracking noise.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
from possession import config, decode, evidence, external, features  # noqa: E402

PROOFS = (1, 2, 3, 4, 6, 8, 12, 16, 25)
TOUCHES = (0.0, 1.0, 2.0, 3.0, 5.0)
NOISES = (0.3, 0.8)


def main():
    matches = {m: external.carry_dead_ball(external.load_match(m))
               for m in external.TRAIN_MATCHES}
    truths = {m: {t: bench.without_touches(match.labels, match.tracking.halves, t)
                  for t in TOUCHES} for m, match in matches.items()}

    print("true possessions per match, by how long a spell must last to count:")
    print("  " + "   ".join("{:g}s: {:.0f}".format(t, np.mean(
        [bench.score(truths[m][t], truths[m][t])["possessions"] for m in matches]))
        for t in TOUCHES))

    saved = {}
    for noise in NOISES:
        rows = {p: {t: [] for t in TOUCHES} for p in PROOFS}
        counts = {p: [] for p in PROOFS}
        for offset, (match_id, match) in enumerate(matches.items()):
            for _, bias, gap in bench.HIDDEN_SCENARIOS:
                version = external.hide_ball(match, offset, stoppage_bias=bias, mean_gap_s=gap,
                                             noise_m=noise)
                votes = evidence.build(features.compute(version.tracking), [])
                for proof in PROOFS:
                    config.P_TEAM_SWITCH = float(proof)
                    labels = decode.decode(votes, version.tracking.halves)
                    for touch in TOUCHES:
                        result = bench.score(labels, truths[match_id][touch])
                        rows[proof][touch].append(result["spell"])
                    counts[proof].append(result["possessions"])

        print("\nspellF1 at {:.1f} m of tracking noise (rows: proof in points, 25 = 1 s of control)".format(
            noise))
        print("{:>7}{:>10}".format("proof", "possess") + "".join(
            "{:>9}".format("{:g}s".format(t)) for t in TOUCHES) + "{:>10}".format("worst"))
        for proof in PROOFS:
            means = [float(np.mean(rows[proof][t])) for t in TOUCHES]
            print("{:>7}{:>10.0f}".format(proof, np.mean(counts[proof])) + "".join(
                "{:>9.3f}".format(m) for m in means) + "{:>10.3f}".format(min(means)))
            saved.setdefault(str(noise), {})[str(proof)] = {
                "possessions": float(np.mean(counts[proof])), "spell": means}

    with open(os.path.join(config.ROOT, "reports", "proof_table.json"), "w") as handle:
        json.dump({"proofs": PROOFS, "touches": TOUCHES, "noises": NOISES, "table": saved}, handle)


if __name__ == "__main__":
    main()
