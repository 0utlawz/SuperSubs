"""Per-frame physical facts about the ball and the players, in metres and seconds.

Nothing here decides a label. Each array answers one plain question -- is the ball over a line,
is someone carrying it, is everybody walking -- and evidence.py turns the answers into votes.
"""

import warnings
from collections import namedtuple

import numpy as np

from . import config
from .data import HOME, AWAY

NOBODY = -1

Features = namedtuple("Features", [
    "ball_seen",     # (T,) ball observed by the tracker on this frame
    "ball_interp",   # (T,) ball filled in across a short gap
    "ball_speed",    # (T,) m/s, NaN where unknown
    "out",           # (T,) ball seen beyond a line
    "control",       # (T,) HOME / AWAY when a player clearly has the ball, else NOBODY
    "near",          # (T,) team nearest the ball within NEAR_RADIUS, else NOBODY
    "slow_score",    # (T,) collective slowness in [-1, 1]; positive means stoppage-like
    "n_tracked",     # (T,) players tracked on this frame
    "ball_xy",       # (T, 2) ball in metres, with short gaps filled; NaN where unknown
    "d_home",        # (T,) metres from the ball to the nearest home player
    "d_away",        # (T,) metres from the ball to the nearest away player
    "rel_home",      # (T,) ball speed relative to that home player
    "rel_away",      # (T,) ball speed relative to that away player
    "closing",       # (T, 2) fastest home / away player closing on the ball, m/s, within APPROACH_M
    "converging",    # (T, 2) home / away players closing on the ball within APPROACH_M
    "around",        # (T, 2) home / away players within AROUND_M of the ball
    "two_nearest",   # (T, 2, 3) the 2 players nearest the ball: metres, cos(bearing), +1 home / -1 away
    "keeper_holds",  # (T,) +1 / -1 when a team's keeper is holding the ball in the box, else 0
])


def graded_mask(n_frames, halves):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in halves:
        mask[first:last + 1] = True
    return mask


def runs(mask):
    """(starts, ends) of each run of True in a 1-D boolean array, ends exclusive."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    steps = np.diff(padded)
    return np.flatnonzero(steps == 1), np.flatnonzero(steps == -1)


def central_velocity(x, half):
    """Time derivative along axis 0 by a centred difference over +/- half frames, per second."""
    v = np.full(x.shape, np.nan)
    v[half:-half] = (x[2 * half:] - x[:-2 * half]) * (config.FPS / (2.0 * half))
    return v


def rolling_nanmean(x, half):
    """Centred mean over +/- half frames, ignoring NaN. NaN where the window has no value."""
    ok = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(x))
    low = np.maximum(0, index - half)
    high = np.minimum(len(x), index + half + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def interpolate_ball(ball, seen, max_gap):
    """Straight-line fill across gaps of at most max_gap frames with the ball seen on both sides."""
    filled = ball.copy()
    interp = np.zeros(len(ball), dtype=bool)
    for start, end in zip(*runs(~seen)):
        if start == 0 or end == len(ball) or end - start > max_gap:
            continue
        frames = np.arange(start, end)
        for axis in range(2):
            filled[start:end, axis] = np.interp(frames, [start - 1, end],
                                                [ball[start - 1, axis], ball[end, axis]])
        interp[start:end] = True
    return filled, interp


def collective_log_speed(tracking):
    """log of the players' mean speed, averaged over +/- SLOW_SMOOTH_HALF_WINDOW frames."""
    tracked = np.isfinite(tracking.players[:, :, 0])
    with np.errstate(invalid="ignore"):
        speeds = np.where(tracked, tracking.players[:, :, 2], np.nan)
        counted = np.isfinite(speeds).sum(1)
        mean_speed = np.where(counted >= config.MIN_TRACKED_FOR_SPEED,
                              np.nansum(speeds, 1) / np.maximum(counted, 1), np.nan)
    smooth = rolling_nanmean(mean_speed, config.SLOW_SMOOTH_HALF_WINDOW)
    return np.log(smooth + 0.05)


def control_radius(speed):
    """How far the ball may be from a player running at speed (m/s) and still be theirs."""
    speed = np.nan_to_num(speed, nan=0.0)  # unknown speed: treat as standing, the strictest
    radius = np.full(speed.shape, config.CONTROL_RADIUS_BY_SPEED[-1][1])
    for upper, value in reversed(config.CONTROL_RADIUS_BY_SPEED):
        radius = np.where(speed < upper, value, radius)
    return radius


def closing_on_ball(pos, player_vel, ball_xy, dist, team_of_slot):
    """For each team: how fast its players close on the ball, and how many of them are doing it.

    Closing speed is the part of a player's velocity that points at the ball, so a player running
    past it scores nothing and one running straight at it scores their full speed.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        toward = ((ball_xy[:, None, 0] - pos[:, :, 0]) * player_vel[:, :, 0]
                  + (ball_xy[:, None, 1] - pos[:, :, 1]) * player_vel[:, :, 1]) / dist
        speed = np.where(np.isfinite(toward) & (dist < config.APPROACH_M), toward, -np.inf)
    closing = np.full((len(pos), 2), np.nan)
    converging = np.zeros((len(pos), 2))
    for side, team in enumerate((HOME, AWAY)):
        fastest = speed[:, team_of_slot == team].max(1)
        closing[:, side] = np.where(np.isfinite(fastest), fastest, np.nan)
        converging[:, side] = (speed[:, team_of_slot == team] > config.APPROACH_SPEED).sum(1)
    return closing, converging


def two_nearest_players(pos, ball_xy, ball_vel, dist, team_of_slot):
    """Distance and bearing to the 2 players nearest the ball, and their teams.

    Bearing is the cosine of the angle between the ball's direction and the line to the player, so
    1 means the ball is heading straight at them, 0 across them and -1 away from them.
    """
    frames = np.arange(len(pos))
    filled = np.where(np.isfinite(dist), dist, np.inf)
    order = np.argsort(filled, axis=1)[:, :2]
    out = np.full((len(pos), 2, 3), np.nan)
    side = np.where(team_of_slot == HOME, 1.0, -1.0)
    ball_speed = np.hypot(ball_vel[:, 0], ball_vel[:, 1])
    for rank in range(2):
        slot = order[:, rank]
        distance = filled[frames, slot]
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = ((pos[frames, slot, 0] - ball_xy[:, 0]) * ball_vel[:, 0]
                      + (pos[frames, slot, 1] - ball_xy[:, 1]) * ball_vel[:, 1]) / (distance * ball_speed)
        out[:, rank, 0] = np.where(np.isfinite(distance), distance, np.nan)
        out[:, rank, 1] = cosine
        out[:, rank, 2] = np.where(np.isfinite(distance), side[slot], np.nan)
    return out


def keeper_holding(pos, ball_xy, ball_speed, ball_seen, tracking):
    """+1 / -1 on frames a keeper holds the ball, for at least KEEPER_HOLD_FRAMES.

    The tracking has no identities, so each team's keeper is whichever of its players is deepest,
    and only counts inside their own penalty box with the ball barely moving at their feet.
    """
    length, width = tracking.pitch
    out = np.zeros(len(pos))
    for _, first, last in tracking.halves:
        block = pos[first:last + 1]
        for team, value in ((HOME, 1.0), (AWAY, -1.0)):
            mask = tracking.team_of_slot == team
            other = tracking.team_of_slot != team
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                goal_at_low_x = np.nanmean(block[:, mask, 0]) < np.nanmean(block[:, other, 0])
            depth = block[:, :, 0] if goal_at_low_x else length - block[:, :, 0]
            depth = np.where(mask[None, :] & np.isfinite(depth), depth, np.inf)
            slot = depth.argmin(1)
            frames = np.arange(len(block))
            gap = np.hypot(block[frames, slot, 0] - ball_xy[first:last + 1, 0],
                           block[frames, slot, 1] - ball_xy[first:last + 1, 1])
            with np.errstate(invalid="ignore"):
                near_goal = (depth[frames, slot] < config.BOX_DEPTH_M) & (
                    np.abs(block[frames, slot, 1] - width / 2) < config.BOX_HALF_WIDTH_M)
                holding = (near_goal & (gap < config.KEEPER_HOLD_M)
                           & (ball_speed[first:last + 1] < config.KEEPER_HOLD_SPEED)
                           & ball_seen[first:last + 1])
            for start, end in zip(*runs(holding)):
                if end - start >= config.KEEPER_HOLD_FRAMES:
                    out[first + start:first + end] = value
    return out


def speed_from_positions(tracking):
    """Players' speed in m/s, worked out from positions.

    Data providers compute speed in their own ways, so a model that reads their speed column learns
    one provider's habits. Working it out here gives the same quantity on every source.
    """
    pos = tracking.players[:, :, :2] * np.array(tracking.pitch)
    velocity = central_velocity(pos, config.VELOCITY_HALF_WINDOW)
    return np.minimum(np.hypot(velocity[:, :, 0], velocity[:, :, 1]), config.MAX_PLAYER_SPEED)


def restart_cue(facts, pitch):
    """The team's two signs of a restart being set up: a still ball, or a slow ball at the touchline.

    Neither fires on the ball in play: under 0.01% of in-play frames on the DFL matches.
    """
    _, width = pitch
    with np.errstate(invalid="ignore"):
        still = facts.ball_seen & (facts.ball_speed < config.STILL_BALL_SPEED)
        y = facts.ball_xy[:, 1]
        at_line = (y < config.TOUCHLINE_M) | (y > width - config.TOUCHLINE_M)
        held_at_line = facts.ball_seen & at_line & (facts.ball_speed < config.THROW_IN_SPEED)
    cue = np.zeros(len(still), dtype=bool)
    for mask in (still, held_at_line):
        for start, end in zip(*runs(mask)):
            if end - start >= config.RESTART_CUE_FRAMES:
                cue[start:end] = True
    return cue


def nearest_of_team(dist, team_mask):
    """(distance, slot) of the nearest tracked player of one team on each frame; inf if none."""
    d = np.where(team_mask[None, :] & np.isfinite(dist), dist, np.inf)
    slot = d.argmin(1)
    return d[np.arange(len(d)), slot], slot


def compute(tracking):
    length, width = tracking.pitch
    scale = np.array([length, width])
    n_frames = len(tracking.ball)
    frames = np.arange(n_frames)

    pos = tracking.players[:, :, :2] * scale
    tracked = np.isfinite(pos[:, :, 0])
    n_tracked = tracked.sum(1)

    ball = tracking.ball[:, :2] * scale
    ball_seen = np.isfinite(ball[:, 0])
    ball_xy, ball_interp = interpolate_ball(ball, ball_seen, config.BALL_INTERP_MAX_GAP)
    ball_have = ball_seen | ball_interp

    ball_vel = central_velocity(ball_xy, config.VELOCITY_HALF_WINDOW)
    ball_speed = np.hypot(ball_vel[:, 0], ball_vel[:, 1])
    player_vel = central_velocity(pos, config.VELOCITY_HALF_WINDOW)

    # -- where the ball is relative to the pitch
    margin = config.OUT_MARGIN
    with np.errstate(invalid="ignore"):
        out = ball_seen & ((ball[:, 0] < -margin) | (ball[:, 0] > length + margin)
                           | (ball[:, 1] < -margin) | (ball[:, 1] > width + margin))

    # -- who is near it, and is anyone carrying it
    dist = np.hypot(pos[:, :, 0] - ball_xy[:, None, 0], pos[:, :, 1] - ball_xy[:, None, 1])
    d_home, slot_home = nearest_of_team(dist, tracking.team_of_slot == HOME)
    d_away, slot_away = nearest_of_team(dist, tracking.team_of_slot == AWAY)

    def relative_speed_and_reach(slot):
        """Ball speed relative to that player, and how far from them the ball may be."""
        vel = player_vel[frames, slot]
        rel = np.hypot(ball_vel[:, 0] - vel[:, 0], ball_vel[:, 1] - vel[:, 1])
        # Where the player's own velocity is unknown (a slot that just appeared), fall back to
        # the ball's speed: a ball that is barely moving is moving with whoever stands on it.
        rel = np.where(np.isfinite(rel), rel, ball_speed)
        return rel, control_radius(np.hypot(vel[:, 0], vel[:, 1]))

    rel_home, reach_home = relative_speed_and_reach(slot_home)
    rel_away, reach_away = relative_speed_and_reach(slot_away)

    with np.errstate(invalid="ignore"):
        home_controls = ((d_home < reach_home) & (rel_home < config.CONTROL_REL_SPEED)
                         & (d_home + config.CONTROL_MARGIN < d_away))
        away_controls = ((d_away < reach_away) & (rel_away < config.CONTROL_REL_SPEED)
                         & (d_away + config.CONTROL_MARGIN < d_home))
    control = np.full(n_frames, NOBODY, dtype=np.int8)
    control[ball_seen & home_controls] = HOME
    control[ball_seen & away_controls] = AWAY

    nearest = np.minimum(d_home, d_away)
    near = np.full(n_frames, NOBODY, dtype=np.int8)
    near_ok = ball_have & (control == NOBODY) & (nearest < config.NEAR_RADIUS)
    near[near_ok] = np.where(d_home[near_ok] <= d_away[near_ok], HOME, AWAY)

    closing, converging = closing_on_ball(pos, player_vel, ball_xy, dist, tracking.team_of_slot)
    two_nearest = two_nearest_players(pos, ball_xy, ball_vel, dist, tracking.team_of_slot)
    keeper_holds = keeper_holding(pos, ball_xy, ball_speed, ball_seen, tracking)
    with np.errstate(invalid="ignore"):
        around = np.stack([(dist[:, tracking.team_of_slot == team] < config.AROUND_M).sum(1)
                           for team in (HOME, AWAY)], axis=1).astype(float)

    # -- collective slowness, against a fixed cut-off set on external matches
    distance = np.log(config.SLOW_CUT_SPEED + 0.05) - collective_log_speed(tracking)
    slow_score = np.clip(np.nan_to_num(distance / config.SLOW_SCALE, nan=0.0), -1.0, 1.0)

    return Features(ball_seen=ball_seen, ball_interp=ball_interp, ball_speed=ball_speed,
                    out=out, control=control, near=near,
                    slow_score=slow_score, n_tracked=n_tracked, ball_xy=ball_xy,
                    d_home=d_home, d_away=d_away, rel_home=rel_home, rel_away=rel_away,
                    closing=closing, converging=converging, around=around,
                    two_nearest=two_nearest, keeper_holds=keeper_holds)
