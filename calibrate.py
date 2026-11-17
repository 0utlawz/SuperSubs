"""Measure how the ball and its carrier behave, on the external DFL matches only.

    python calibrate.py

The rules for choosing each setting were agreed before these numbers existed, and carried over
from the team's calibration session: the "has the ball" reach is the 90th percentile of the
carrier's distance in each running-speed band, and "moving with the ball" is the 85th percentile
of the relative speed. Nothing here reads the competition match.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from possession import config, external, features  # noqa: E402

HALF = config.VELOCITY_HALF_WINDOW

# Carrier speed bands, m/s: jogging, running, fast running, sprinting.
SPEED_BANDS = ((0.0, 3.0, "jog     under 3 m/s"), (3.0, 5.0, "run     3-5 m/s"),
               (5.0, 7.0, "fast    5-7 m/s"), (7.0, np.inf, "sprint  7+ m/s"))


def carrier_samples(match):
    """(distance, relative speed, carrier speed, ball speed) on frames a player runs with the ball.

    Picked from the truth: ball in play, nearest player is on the team in possession, the same
    player stays nearest for a second or more, and ball and player move the same way at the same
    pace. Distance is not used to pick them, so the distances are not circular.
    """
    tracking = match.tracking
    scale = np.array(tracking.pitch)
    frames = np.arange(len(tracking.ball))
    pos = tracking.players[:, :, :2] * scale
    ball = tracking.ball[:, :2] * scale
    ball_vel = features.central_velocity(ball, HALF)
    ball_speed = np.hypot(ball_vel[:, 0], ball_vel[:, 1])

    dist = np.hypot(pos[:, :, 0] - ball[:, None, 0], pos[:, :, 1] - ball[:, None, 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest = dist.argmin(1)
    d_near = dist[frames, nearest]
    player_vel = features.central_velocity(pos, HALF)[frames, nearest]
    player_speed = np.hypot(player_vel[:, 0], player_vel[:, 1])
    rel = np.hypot(ball_vel[:, 0] - player_vel[:, 0], ball_vel[:, 1] - player_vel[:, 1])

    side = np.where(tracking.team_of_slot[nearest] == 0, "H", "A")
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (ball_vel * player_vel).sum(1) / (ball_speed * player_speed)
        ratio = ball_speed / player_speed
        in_step = ((match.labels == side) & np.isfinite(d_near) & (player_speed > 1.5)
                   & (cosine > 0.9) & (ratio > 0.67) & (ratio < 1.5))

    same = np.concatenate(([False], (nearest[1:] == nearest[:-1]) & in_step[1:] & in_step[:-1]))
    held = np.zeros(len(same), dtype=bool)
    for start, end in zip(*features.runs(same)):
        if end - start + 1 >= config.FPS:
            held[start - 1:end] = True
    return d_near[held], rel[held], player_speed[held], ball_speed[held]


def main():
    samples = [carrier_samples(external.load_match(m)) for m in external.TRAIN_MATCHES]
    distance, relative, speed, _ = (np.concatenate(parts) for parts in zip(*samples))
    print("{} carrying frames across {} matches\n".format(len(distance), len(samples)))

    print("{:<22}{:>8}{:>7}{:>7}{:>7}{:>7}{:>7}   relative speed p85".format(
        "carrier speed", "frames", "p50", "p80", "p85", "p90", "p95"))
    reaches = []
    for low, high, name in SPEED_BANDS:
        band = (speed >= low) & (speed < high)
        row = np.percentile(distance[band], (50, 80, 85, 90, 95))
        print("{:<22}{:>8}".format(name, int(band.sum()))
              + "".join("{:>7.2f}".format(v) for v in row)
              + "{:>12.2f}".format(np.percentile(relative[band], 85)))
        reaches.append((high, round(float(row[3]), 2)))

    print("\nCONTROL_RADIUS_BY_SPEED = {}".format(tuple(reaches)))
    print("CONTROL_REL_SPEED = {:.2f}   (85th percentile, all speeds)".format(
        np.percentile(relative, 85)))

    fastest = np.percentile(speed, 99)
    print("carriers' own speed, 99th percentile: {:.2f} m/s".format(fastest))

    durations = []
    for match_id in external.TRAIN_MATCHES:
        match = external.load_match(match_id)
        for _, first, last in match.tracking.halves:
            dead = match.labels[first:last + 1] == "D"
            starts, ends = features.runs(dead)
            durations.append(ends - starts)
    seconds = np.concatenate(durations) / config.FPS
    print("\nstoppages: n={}  shortest 1% {:.1f} s  5% {:.1f} s  median {:.1f} s  95% {:.1f} s".format(
        len(seconds), *np.percentile(seconds, (1, 5, 50, 95))))
    print("share under the 2 s minimum: {:.1f}%".format(100 * np.mean(seconds < 2)))


if __name__ == "__main__":
    main()
