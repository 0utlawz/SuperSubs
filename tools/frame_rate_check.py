"""Why do the Clasicos go the other way? Recreate their conditions, keeping the real answers.

    MTAH_CANDIDATE=v5 python tools/frame_rate_check.py

The Clasicos differ from the DFL matches in two ways at once: rougher conditions, and a weak answer
key. This keeps DFL's official answers on the held-out matches and changes only the conditions: 10
frames a second, then a TV camera and cuts as well. The last column scores against a coarser truth
with spells under 3 s removed, closer to the long spells of SkillCorner's flag. No setting changes.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
from possession import config, external, features  # noqa: E402

COARSE_S = 3.0
SOURCE_STEP = config.FPS / 10.0
CUT_SHARE = 0.26  # the Clasicos have no tracking on about 26% of play
CUT_MEAN_S = 15.0


def resampled(values, n_frames):
    """Sample at 10 frames a second, then draw straight lines back up to 25, as the Clasicos are."""
    picked = np.round(np.arange(0, n_frames, SOURCE_STEP)).astype(int)
    picked = picked[picked < n_frames]
    samples = values[picked]
    position = np.arange(n_frames) / SOURCE_STEP
    low = np.minimum(position.astype(int), len(samples) - 2)
    weight = (position - low).reshape((-1,) + (1,) * (values.ndim - 1))
    return (1 - weight) * samples[low] + weight * samples[low + 1]


def at_10_fps(tracking):
    n_frames = len(tracking.ball)
    players = tracking.players.copy()
    ball = tracking.ball.copy()
    players[:, :, :2] = resampled(players[:, :, :2], n_frames)
    ball[:, :2] = resampled(ball[:, :2], n_frames)
    scale = np.array(tracking.pitch)
    velocity = features.central_velocity(players[:, :, :2] * scale, config.VELOCITY_HALF_WINDOW)
    players[:, :, 2] = np.hypot(velocity[:, :, 0], velocity[:, :, 1])
    return tracking._replace(players=players, ball=ball)


def with_cuts(tracking, seed):
    """TV cuts: stretches with no tracking at all."""
    rng = np.random.default_rng(seed)
    players, ball = tracking.players.copy(), tracking.ball.copy()
    n_frames = len(ball)
    cut = np.zeros(n_frames, dtype=bool)
    while cut.mean() < CUT_SHARE:
        start = int(rng.integers(n_frames))
        cut[start:start + max(int(rng.exponential(CUT_MEAN_S) * config.FPS), 1)] = True
    players[cut] = np.nan
    ball[cut] = np.nan
    return tracking._replace(players=players, ball=ball)


def main():
    candidates = bench.load_candidates()
    bundle = bench.load_bundle()
    levels = ("25 fps", "10 fps", "10 fps + TV camera + cuts")
    results = {}
    for offset, match_id in enumerate(external.TEST_MATCHES):
        match = external.carry_dead_ball(external.load_match(match_id))
        halves = match.tracking.halves
        touchless = bench.without_touches(match.labels, halves)
        coarse = bench.without_touches(match.labels, halves, COARSE_S)
        for gap_index, gap in enumerate((3.0, 12.0, 48.0)):
            seed = 4000 + offset * 10 + gap_index
            plain = external.hide_ball(match, seed, stoppage_bias=4.0, mean_gap_s=gap, noise_m=0.3)
            camera = external.hide_ball(match, seed, stoppage_bias=4.0, mean_gap_s=gap, noise_m=0.5,
                                        share=0.15, camera_half_m=20.0, id_swap_s=30.0)
            trackings = (plain.tracking, at_10_fps(plain.tracking),
                         at_10_fps(with_cuts(camera.tracking, seed)))
            for level, tracking in zip(levels, trackings):
                methods = [("baseline", bench.baseline_labels(tracking)),
                           ("version1", bench.pipeline_labels(tracking)),
                           ("version2", bench.pipeline_v2_labels(tracking, bundle))]
                for name, candidate in candidates:
                    methods.append((name, bench.pipeline_v2_labels(tracking, candidate)))
                for method, labels in methods:
                    raw = bench.score(labels, match.labels)
                    results.setdefault((level, method), []).append(
                        (raw["spell"], bench.score(labels, touchless)["spell"],
                         bench.score(labels, coarse)["spell"], raw["macro"], raw["possessions"]))

    print("{:<28}{:<11}{:>9}{:>12}{:>12}{:>9}{:>9}".format(
        "conditions", "method", "spellF1", "no-touches", "coarse 3 s", "macroF1", "possess"))
    for (level, method), rows in results.items():
        print("{:<28}{:<11}{:>9.3f}{:>12.3f}{:>12.3f}{:>9.3f}{:>9.0f}".format(
            level, method, *np.mean(rows, axis=0)))


if __name__ == "__main__":
    main()
