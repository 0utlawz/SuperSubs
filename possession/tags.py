"""Smart Tagging's set-piece tags, as votes around each restart.

A set-piece tag says a restart happens around here. Before the restart the ball is dead; after it,
the team taking it has the ball. Smart Tagging exists only for the competition match, so its weights
cannot be tuned there and cannot be tuned on real tags anywhere else. Instead imitate() builds tags
from the official restarts of the DFL matches, shaped like the real ones from what the organisers
publish: how many of each type, a window of about 5 s around the restart, accurate to about a second,
and no team on throw-ins.
"""

import numpy as np

from . import config
from .data import HOME, AWAY, Tag
from .evidence import A, D, H
from .features import runs

# Share of each restart type that gets a tag: the published tag counts for the competition match
# (26 throw-ins, 12 corners, 10 goal kicks, 5 free kicks, 6 kick-offs) against how many of each a
# match has on the DFL data (about 44, 8, 15, 31 and 3).
TAG_SHARE = {"Throw-in": 0.6, "Corner": 0.9, "Goal kick": 0.65, "Free kick / other": 0.16,
             "Kick-off": 0.9}
WINDOW_S = 5.0
TIMING_ERROR_S = 1.0
FALSE_TAG_SHARE = 0.1

# Metres from the centre spot, for naming a restart from where the ball is when play resumes.
KICK_OFF_RADIUS = 3.0
CORNER_REACH = 3.0
TOUCHLINE_REACH = 2.0
GOAL_AREA_DEPTH = 6.5
GOAL_AREA_HALF_WIDTH = 11.0


def restart_type(x, y, length, width):
    if np.hypot(x, y) <= KICK_OFF_RADIUS:
        return "Kick-off"
    near_goal_line = abs(x) >= length / 2 - CORNER_REACH
    near_touchline = abs(y) >= width / 2 - TOUCHLINE_REACH
    if near_goal_line and abs(y) >= width / 2 - CORNER_REACH:
        return "Corner"
    if near_touchline:
        return "Throw-in"
    if abs(x) >= length / 2 - GOAL_AREA_DEPTH and abs(y) <= GOAL_AREA_HALF_WIDTH:
        return "Goal kick"
    return "Free kick / other"


def imitate(match, seed, false_share=FALSE_TAG_SHARE, timing_error_s=TIMING_ERROR_S):
    """Tags for a DFL match, shaped like the competition match's set-piece tags."""
    rng = np.random.default_rng(seed)
    tracking = match.tracking
    length, width = tracking.pitch
    ball = tracking.ball[:, :2] * np.array(tracking.pitch)
    labels = match.labels
    half_width = int(WINDOW_S * config.FPS / 2)

    tags = []
    for _, first, last in tracking.halves:
        starts, ends = runs(labels[first:last + 1] == "D")
        for end in ends + first:
            if end > last:
                continue
            kind = restart_type(ball[end, 0] - length / 2, ball[end, 1] - width / 2, length, width)
            if rng.random() > TAG_SHARE[kind]:
                continue
            centre = int(end + rng.normal(0, timing_error_s) * config.FPS)
            team = None if kind == "Throw-in" else (HOME if labels[end] == "H" else AWAY)
            tags.append(Tag("set pieces", centre - half_width, centre + half_width, team, kind, None))

    alive = np.flatnonzero(np.isin(labels, ("H", "A")))
    for _ in range(int(round(false_share * len(tags)))):
        centre = int(rng.choice(alive))
        tags.append(Tag("set pieces", centre - half_width, centre + half_width,
                        int(rng.integers(2)), "false", None))
    return tags


def add_votes(evidence, tags, weight_dead, lookback_s, weight_team, guard_s=0.5):
    """Vote dead ball before each tagged restart, and for the team taking it just after.

    The restart is taken to be the middle of the tag. guard_s leaves its timing error unclaimed.
    """
    n_frames = len(evidence)
    guard = int(guard_s * config.FPS)
    lookback = int(lookback_s * config.FPS)
    for tag in tags:
        middle = (tag.start + tag.end) // 2
        evidence[max(0, middle - guard - lookback):max(0, middle - guard), D] += weight_dead
        if tag.team is not None:
            column = H if tag.team == HOME else A
            evidence[min(n_frames, middle + guard):min(n_frames, tag.end), column] += weight_team
    return evidence
