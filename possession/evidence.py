"""Turn the per-frame facts and the Smart Tagging into votes for each state on each frame.

E[t, s] is how much frame t supports state s (H, A or D), in "frames of clean control". The
decoder adds these up along a path and pays a fixed price every time the state changes, so a vote
only matters when enough of it accumulates to pay for a change.

A vote that does not discriminate between H and A -- "the ball is moving fast, so it is in play" --
is added to both, so it argues for play over dead ball without taking a side.
"""

import numpy as np

from . import config
from .data import HOME, AWAY
from .features import NOBODY

H, A, D = 0, 1, 2
STATES = "HAD"


def _team_column(team):
    return H if team == HOME else A


def _add_tags(evidence, tags):
    n_frames = len(evidence)

    def span(start, end):
        return slice(max(0, start), min(n_frames, end))

    for tag in tags:
        if tag.team is None:
            continue
        if tag.code in config.TAG_TEAM_HAS_BALL:
            evidence[span(tag.start, tag.end), _team_column(tag.team)] += config.W_TAG_PHASE
        elif tag.code in config.TAG_TEAM_LACKS_BALL:
            other = AWAY if tag.team == HOME else HOME
            evidence[span(tag.start, tag.end), _team_column(other)] += config.W_TAG_PHASE
        elif tag.code == "set pieces":
            # The tagged window surrounds the restart. Its second half is after the ball is
            # played, when the team taking it has the ball.
            middle = (tag.start + tag.end) // 2
            evidence[span(middle, tag.end), _team_column(tag.team)] += config.W_TAG_SET_PIECE

    # Between a goal and the kick-off that restarts play, the ball is dead.
    kickoffs = sorted((tag for tag in tags
                       if tag.code == "set pieces" and tag.kind.startswith("kick off")),
                      key=lambda tag: tag.start)
    for goal in (tag for tag in tags if tag.code == "goals"):
        restart = next((k for k in kickoffs if k.start >= goal.end), None)
        if restart is not None and restart.start - goal.end < 3 * 60 * config.FPS:
            middle = (restart.start + restart.end) // 2
            evidence[span(goal.end, middle), D] += config.W_GOAL_DEAD


def build(features, tags):
    """(T, 3) evidence for H, A, D on every video frame."""
    n_frames = len(features.control)
    evidence = np.zeros((n_frames, 3))

    # Who has the ball.
    for team in (HOME, AWAY):
        evidence[features.control == team, _team_column(team)] += config.W_CONTROL
        evidence[features.near == team, _team_column(team)] += config.W_NEAR

    # Whether it is in play.
    with np.errstate(invalid="ignore"):
        moving = features.ball_seen & ~features.out & (features.ball_speed > config.MOVING_SPEED)
    evidence[moving, H] += config.W_MOVING
    evidence[moving, A] += config.W_MOVING
    evidence[features.out, D] += config.W_OUT
    evidence[:, D] += config.W_SLOW * features.slow_score

    _add_tags(evidence, tags)
    return evidence
