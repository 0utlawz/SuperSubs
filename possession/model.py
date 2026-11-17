"""Version 2: votes learned from external matches, in place of the hand-written ones.

Two models, trained only on the DFL matches, feed the same decoder as version 1:

  team model   Given the ball is in play, does the home team have it? It sees the ball when there
               is one, and otherwise the players' shape and the clues around the gap.
  dead model   Is the ball dead? It sees the players only, never the ball, because the DFL ball is
               synthetic during stoppages and a model could learn that instead of football.

Every feature can be computed from the competition tracking: positions, speeds and team membership,
with no player identity and no ball height.
"""

import pickle
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config, decode, features, tags as set_pieces
from .data import AWAY, HOME
from .evidence import A, D, H

# Half-widths in frames for summing control and nearest-player signals over time.
SIGNAL_WINDOWS = (10, 25, 50, 125)
# Half-widths in frames for averaging team shape over time.
SHAPE_WINDOWS = (25, 100)
# Frames either side for forward velocity, and for spotting everyone slowing down or speeding up.
VELOCITY_HALF = 12
SPEED_SHIFT = 50
MAX_AGE_S = 60.0
FAST_SPEED = 4.0  # m/s: a player clearly running, not walking back to position

# A touch shows up as a sudden change in the ball's velocity, measured across this many frames either
# side (0.32 s in all). 9 m/s is where credited touches best match the team in possession on the DFL
# matches, and it gives a realistic count against their 886 official plays per match; 3 m/s fires
# on ordinary ball movement.
TOUCH_SHIFT = 4
TOUCH_CHANGE = 9.0
TOUCH_REACH = 4.0  # m: the nearest player must be this close for the touch to be credited

APPROACH_SHIFTS = (6, 12, 25)  # frames over which a change in distance to the ball is measured
RELIEF_MAX_M = 15.0  # a team this much closer than the other has the ball to itself
LONG_BALL_M = 20.0  # a ball travelling this far between touches is a long ball or a cross
TURN_SHIFTS = (25, 50, 75)  # frames either side for spotting both teams changing direction

CONTROL_LENGTHS = (15, 25)  # frames on the ball: the team's rule, and a full second

GATHER_STRIDE = 5
GATHER_AHEAD_S = 1.0  # players are projected this far ahead
GATHER_RADIUS_M = 8.0  # players within this distance of each other count as one group

PROBABILITY_FLOOR = 0.005


def _make_model(seed=0):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)


def attack_sign(tracking):
    """+1 on frames where the home side attacks towards x = 1, -1 where it attacks towards x = 0.

    Over a half each team averages nearer its own goal, so comparing the two means settles it.
    """
    sign = np.ones(len(tracking.ball))
    home = tracking.team_of_slot == HOME
    x = tracking.players[:, :, 0]
    for _, first, last in tracking.halves:
        block = x[first:last + 1]
        if np.nanmean(block[:, home]) > np.nanmean(block[:, ~home]):
            sign[first:last + 1] = -1.0
    return sign


def _rolling_sum(values, half):
    total = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(len(values))
    return total[np.minimum(len(values), index + half + 1)] - total[np.maximum(0, index - half)]


def _last_and_next(event, values):
    """For each frame: value at, and seconds since, the last event; and the same for the next."""
    index = np.arange(len(event))
    last = np.maximum.accumulate(np.where(event, index, -1))
    upcoming = np.minimum.accumulate(np.where(event, index, len(event))[::-1])[::-1]
    has_last, has_next = last >= 0, upcoming < len(event)

    last_value = np.where(has_last, values[np.maximum(last, 0)], 0.0)
    next_value = np.where(has_next, values[np.minimum(upcoming, len(event) - 1)], 0.0)
    since = np.where(has_last, (index - last) / config.FPS, MAX_AGE_S)
    until = np.where(has_next, (upcoming - index) / config.FPS, MAX_AGE_S)
    return last_value, np.minimum(since, MAX_AGE_S), next_value, np.minimum(until, MAX_AGE_S)


def _shape_columns(tracking, derived_speed=False):
    """Team shape on every frame, with the home side attacking towards positive x."""
    length, width = tracking.pitch
    sign = attack_sign(tracking)[:, None]
    x = (tracking.players[:, :, 0] - 0.5) * length * sign
    y = (tracking.players[:, :, 1] - 0.5) * width
    speed = (features.speed_from_positions(tracking) if derived_speed
             else tracking.players[:, :, 2])
    forward = features.central_velocity(x, VELOCITY_HALF)
    home = tracking.team_of_slot == HOME
    away = tracking.team_of_slot == AWAY

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        columns = {
            "cx_h": np.nanmean(x[:, home], 1), "cx_a": np.nanmean(x[:, away], 1),
            "cy_h": np.nanmean(y[:, home], 1), "cy_a": np.nanmean(y[:, away], 1),
            "sx_h": np.nanstd(x[:, home], 1), "sx_a": np.nanstd(x[:, away], 1),
            "sy_h": np.nanstd(y[:, home], 1), "sy_a": np.nanstd(y[:, away], 1),
            "front_h": np.nanmax(x[:, home], 1), "front_a": np.nanmin(x[:, away], 1),
            "back_h": np.nanmin(x[:, home], 1), "back_a": np.nanmax(x[:, away], 1),
            # Medians: a track that changes hands makes one player appear to jump across the pitch.
            "vx_h": np.nanmedian(forward[:, home], 1), "vx_a": np.nanmedian(forward[:, away], 1),
            "speed_h": np.nanmean(speed[:, home], 1), "speed_a": np.nanmean(speed[:, away], 1),
        }
    columns["count_h"] = np.isfinite(x[:, home]).sum(1).astype(float)
    columns["count_a"] = np.isfinite(x[:, away]).sum(1).astype(float)

    derived = {"gap": columns["cx_h"] - columns["cx_a"],
               "depth": columns["sx_h"] - columns["sx_a"],
               "breadth": columns["sy_h"] - columns["sy_a"],
               "drive": columns["vx_h"] + columns["vx_a"],
               "speed_gap": columns["speed_h"] - columns["speed_a"]}
    columns.update(derived)
    for name in ("gap", "depth", "breadth", "drive", "vx_h", "vx_a", "speed_gap"):
        for half in SHAPE_WINDOWS:
            columns["{}_m{}".format(name, half)] = features.rolling_nanmean(columns[name], half)
    return columns


def _matrix(columns):
    names = sorted(columns)
    return np.stack([columns[name] for name in names], axis=1).astype(np.float32), names


def _touch_events(facts):
    """(velocity change, team nearest the ball as +1 / -1, touch mask) on every frame."""
    velocity = features.central_velocity(facts.ball_xy, config.VELOCITY_HALF_WINDOW)
    change = np.full(len(velocity), np.nan)
    shift = TOUCH_SHIFT
    change[shift:-shift] = np.hypot(velocity[2 * shift:, 0] - velocity[:-2 * shift, 0],
                                    velocity[2 * shift:, 1] - velocity[:-2 * shift, 1])
    change = np.where(facts.ball_seen, change, np.nan)

    with np.errstate(invalid="ignore"):
        nearest = np.minimum(facts.d_home, facts.d_away)
        side = np.where(facts.d_home <= facts.d_away, 1.0, -1.0)
        peak = np.zeros(len(change), dtype=bool)
        filled = np.nan_to_num(change, nan=0.0)
        peak[shift:-shift] = (filled[shift:-shift] >= filled[:-2 * shift]) & (
            filled[shift:-shift] >= filled[2 * shift:])
        touch = peak & (filled > TOUCH_CHANGE) & (nearest < TOUCH_REACH)
    return change, side, touch


def _touch_columns(tracking, facts):
    """Who touched the ball, and when: the clue that separates one-touch play and interceptions.

    Control needs a player to stay on the ball, so a first-time pass or a quick interception barely
    registers. A touch does: the ball's path bends or its speed jumps, next to someone.
    """
    change, side, touch = _touch_events(facts)
    signed = np.where(touch, side, 0.0)
    columns = {"kick": change, "kick_side": np.nan_to_num(change, nan=0.0) * side}
    for half in SIGNAL_WINDOWS:
        columns["touch_{}".format(half)] = _rolling_sum(signed, half)

    index = np.arange(len(change))
    events = np.flatnonzero(touch)
    if len(events) < 2:
        for name in ("last", "last2", "next", "next2", "since", "until"):
            columns["touch_{}".format(name)] = np.zeros(len(change))
        return columns

    before = np.searchsorted(events, index, side="right") - 1
    after = before + 1
    sides = signed[events]

    def pick(position, fallback):
        inside = (position >= 0) & (position < len(events))
        return np.where(inside, sides[np.clip(position, 0, len(events) - 1)], fallback)

    columns["touch_last"], columns["touch_last2"] = pick(before, 0.0), pick(before - 1, 0.0)
    columns["touch_next"], columns["touch_next2"] = pick(after, 0.0), pick(after + 1, 0.0)
    since = (index - events[np.clip(before, 0, len(events) - 1)]) / config.FPS
    until = (events[np.clip(after, 0, len(events) - 1)] - index) / config.FPS
    columns["touch_since"] = np.where(before >= 0, np.minimum(since, MAX_AGE_S), MAX_AGE_S)
    columns["touch_until"] = np.where(after < len(events), np.minimum(until, MAX_AGE_S), MAX_AGE_S)
    return columns


def _contact_columns(columns, facts):
    """Contact to contact: did the same team keep it, and who was there when the ball arrived?

    A pass is two touches by one team. One opposing touch followed by the first team again is a
    deflection; an opposing touch followed by that team again is a real change. A long ball is
    judged where it comes down: a player alone there has it, a contested one may not.
    """
    n_frames = len(facts.control)
    _, side, touch = _touch_events(facts)
    events = np.flatnonzero(touch)
    names = ("chain_same", "run_last", "run_next", "margin_last", "margin_next", "landing_home",
             "landing_away", "flight_m", "flight_s", "long_ball")
    if len(events) < 3:
        return {name: np.zeros(n_frames) for name in names}

    sides = side[events]
    run_back, run_on = np.ones(len(events)), np.ones(len(events))
    for i in range(1, len(events)):
        run_back[i] = run_back[i - 1] + 1 if sides[i] == sides[i - 1] else 1
    for i in range(len(events) - 2, -1, -1):
        run_on[i] = run_on[i + 1] + 1 if sides[i] == sides[i + 1] else 1

    d_home = np.where(np.isfinite(facts.d_home), facts.d_home, RELIEF_MAX_M)
    d_away = np.where(np.isfinite(facts.d_away), facts.d_away, RELIEF_MAX_M)
    margin = np.clip(d_away - d_home, -RELIEF_MAX_M, RELIEF_MAX_M)[events]
    where = facts.ball_xy[events]

    index = np.arange(n_frames)
    before = np.searchsorted(events, index, side="right") - 1
    after = before + 1
    has_before, has_after = before >= 0, after < len(events)

    def at(values, position, present):
        return np.where(present, values[np.clip(position, 0, len(events) - 1)], 0.0)

    last, upcoming = at(sides, before, has_before), at(sides, after, has_after)
    both = has_before & has_after
    gap = where[np.clip(after, 0, len(events) - 1)] - where[np.clip(before, 0, len(events) - 1)]
    flight = np.where(both, np.hypot(gap[:, 0], gap[:, 1]), 0.0)
    return {
        "chain_same": np.where(last == upcoming, last, 0.0),
        "run_last": at(run_back, before, has_before) * last,
        "run_next": at(run_on, after, has_after) * upcoming,
        "margin_last": at(margin, before, has_before),
        "margin_next": at(margin, after, has_after),
        "landing_home": at(facts.around[events, 0], after, has_after),
        "landing_away": at(facts.around[events, 1], after, has_after),
        "flight_m": flight,
        "flight_s": np.where(both, (events[np.clip(after, 0, len(events) - 1)]
                                    - events[np.clip(before, 0, len(events) - 1)]) / config.FPS, 0.0),
        "long_ball": (flight > LONG_BALL_M).astype(float),
    }


def _reappear_columns(facts):
    """Which team was closer when the ball vanished, and when it came back."""
    gap = np.where(np.isfinite(facts.d_home) & np.isfinite(facts.d_away),
                   np.clip(facts.d_away - facts.d_home, -RELIEF_MAX_M, RELIEF_MAX_M), 0.0)
    last, _, upcoming, _ = _last_and_next(facts.ball_seen, gap)
    return {"margin_vanish": last, "margin_return": upcoming}


def _turnaround_columns(columns):
    """When possession flips, both teams change direction together within a second or two."""
    drive = np.nan_to_num(columns["drive_m25"], nan=0.0)
    middle = np.nan_to_num((columns["cx_h"] + columns["cx_a"]) / 2, nan=0.0)
    out = {}
    for shift in TURN_SHIFTS:
        out["turn_{}".format(shift)] = np.roll(drive, -shift) - np.roll(drive, shift)
        out["drive_next_{}".format(shift)] = np.roll(drive, -shift)
        out["drive_past_{}".format(shift)] = np.roll(drive, shift)
        out["shift_next_{}".format(shift)] = np.roll(middle, -shift) - middle
        out["shift_past_{}".format(shift)] = middle - np.roll(middle, shift)
    return out


def _expert_columns(facts):
    """The team's rules: judge 10 frames after a touch, read the ball's bearing, hold for 15 frames.

    1. A change of team is judged a fixed time after the first touch.
    2. A loose ball goes to the player it is heading for, by distance and bearing to the nearest 2.
    3. After a header or a save, what counts is who collects it.
    4. A keeper who holds the ball has it for their team.
    5. A touch by the other side between two team-mates changes nothing.
    6. Control means staying on the ball, and only then does it count.
    """
    n_frames = len(facts.control)
    delay = config.JUDGE_AFTER_FRAMES
    _, side, touch = _touch_events(facts)

    distance, cosine, team = (facts.two_nearest[:, :, i] for i in range(3))
    columns = {"near1_d": distance[:, 0], "near1_bearing": cosine[:, 0], "near1_team": team[:, 0],
               "near2_d": distance[:, 1], "near2_bearing": cosine[:, 1], "near2_team": team[:, 1]}

    # Rule 2: who the ball is heading for. Closer and straighter counts for more.
    pull = np.maximum(np.nan_to_num(cosine, nan=0.0), 0.0) / (np.nan_to_num(distance, nan=1e6) + 1.0)
    heading = np.nan_to_num((pull * team).sum(1), nan=0.0)
    columns["heading"] = heading
    for half in (delay, 2 * delay):
        columns["heading_{}".format(half)] = _rolling_sum(heading, half)

    # Rule 1: who is closest once the judgment is due, after the last touch and the next one.
    margin = np.where(np.isfinite(facts.d_home) & np.isfinite(facts.d_away),
                      np.clip(facts.d_away - facts.d_home, -RELIEF_MAX_M, RELIEF_MAX_M), 0.0)
    judged = np.zeros(n_frames)
    judged[:-delay] = margin[delay:]
    events = np.flatnonzero(touch)
    if len(events) >= 3:
        before = np.searchsorted(events, np.arange(n_frames), side="right") - 1
        after = before + 1
        sides = side[events]

        def at(values, position):
            inside = (position >= 0) & (position < len(events))
            return np.where(inside, values[np.clip(position, 0, len(events) - 1)], 0.0)

        columns["judged_last"] = at(judged[events], before)
        columns["judged_next"] = at(judged[events], after)
        # Rule 5: the same team on both sides of a single opposing touch.
        first, middle, third = at(sides, before - 1), at(sides, before), at(sides, after)
        columns["deflected"] = np.where((first == third) & (middle != first), first, 0.0)
    else:
        for name in ("judged_last", "judged_next", "deflected"):
            columns[name] = np.zeros(n_frames)

    # Rule 6 and rule 3: a team staying on the ball for long enough, and who does so next.
    on_ball = np.where(facts.control == HOME, 1.0, np.where(facts.control == AWAY, -1.0, 0.0))
    near = np.where(facts.near == HOME, 1.0, np.where(facts.near == AWAY, -1.0, 0.0))
    on_ball = np.where(on_ball != 0, on_ball, near)
    held = np.zeros(n_frames)
    for value in (1.0, -1.0):
        for start, end in zip(*features.runs(on_ball == value)):
            if end - start >= config.CONTROL_FRAMES:
                held[start:end] = value
    last, since, upcoming, until = _last_and_next(held != 0, held)
    columns.update({"held": held, "held_last": last, "held_since": since, "held_next": upcoming,
                    "held_until": until})

    # Rule 4: a keeper holding the ball.
    last, since, upcoming, until = _last_and_next(facts.keeper_holds != 0, facts.keeper_holds)
    columns.update({"keeper": facts.keeper_holds, "keeper_last": last, "keeper_since": since,
                    "keeper_next": upcoming, "keeper_until": until})
    return columns


def _clear_control_columns(facts):
    """The team's two-part test of control: on the ball, and nobody close enough to take it."""
    n_frames = len(facts.control)
    on_ball = np.where(facts.control == HOME, 1.0, np.where(facts.control == AWAY, -1.0, 0.0))
    near = np.where(facts.near == HOME, 1.0, np.where(facts.near == AWAY, -1.0, 0.0))
    on_ball = np.where(on_ball != 0, on_ball, near)
    opponent = np.where(on_ball > 0, facts.d_away, np.where(on_ball < 0, facts.d_home, np.nan))
    with np.errstate(invalid="ignore"):
        clear = on_ball != 0
        clear = clear & (np.nan_to_num(opponent, nan=0.0, posinf=99.0) >= config.TACKLE_M)
    columns = {"opponent_m": np.where(np.isfinite(opponent), opponent, np.nan)}
    for frames in CONTROL_LENGTHS:
        held = np.zeros(n_frames)
        for value in (1.0, -1.0):
            for start, end in zip(*features.runs((on_ball == value) & clear)):
                if end - start >= frames:
                    held[start:end] = value
        last, since, upcoming, until = _last_and_next(held != 0, held)
        columns.update({"clear_{}".format(frames): held, "clear_{}_last".format(frames): last,
                        "clear_{}_since".format(frames): since,
                        "clear_{}_next".format(frames): upcoming,
                        "clear_{}_until".format(frames): until})
    return columns


def _gather_columns(tracking):
    """Where the players are gathering, for when the ball cannot be seen.

    With the ball off-screen, a good judge watches where players are heading and who is contesting
    it. Each player is projected 1 s ahead, and the busiest spot on the pitch is taken to be where
    the ball is, or is about to be. Sampled every GATHER_STRIDE frames and held in between.
    """
    length, width = tracking.pitch
    n_frames = len(tracking.ball)
    sign = attack_sign(tracking)[:, None]
    x = (tracking.players[:, :, 0] - 0.5) * length * sign
    y = (tracking.players[:, :, 1] - 0.5) * width
    vx = features.central_velocity(x, VELOCITY_HALF)
    vy = features.central_velocity(y, VELOCITY_HALF)
    ahead_x = x + np.where(np.isfinite(vx), vx, 0.0) * GATHER_AHEAD_S
    ahead_y = y + np.where(np.isfinite(vy), vy, 0.0) * GATHER_AHEAD_S

    frames = np.arange(0, n_frames, GATHER_STRIDE)
    px, py = ahead_x[frames].astype(np.float32), ahead_y[frames].astype(np.float32)
    home = tracking.team_of_slot == HOME
    gap = np.hypot(px[:, :, None] - px[:, None, :], py[:, :, None] - py[:, None, :])
    close = np.nan_to_num(gap, nan=np.inf) < GATHER_RADIUS_M
    busiest = close.sum(2).argmax(1)
    rows = np.arange(len(frames))
    group = close[rows, busiest]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        centre_x = np.nanmean(np.where(group, px, np.nan), 1)
        centre_y = np.nanmean(np.where(group, py, np.nan), 1)
    to_centre = np.nan_to_num(np.hypot(px - centre_x[:, None], py - centre_y[:, None]), nan=np.inf)
    columns = {
        "gather_x": centre_x, "gather_y": np.abs(centre_y),
        "gather_home": group[:, home].sum(1).astype(np.float32),
        "gather_away": group[:, ~home].sum(1).astype(np.float32),
        "gather_d_home": to_centre[:, home].min(1), "gather_d_away": to_centre[:, ~home].min(1),
    }
    columns["gather_gap"] = columns["gather_d_away"] - columns["gather_d_home"]
    columns["gather_count_gap"] = columns["gather_home"] - columns["gather_away"]
    held = np.minimum(np.arange(n_frames) // GATHER_STRIDE, len(frames) - 1)
    return {name: np.where(np.isfinite(values), values, np.nan)[held]
            for name, values in columns.items()}


def _approach_columns(facts):
    """Which team is coming to the ball: the nearest player's distance changing, and who is closing."""
    d_home = np.where(np.isfinite(facts.d_home), facts.d_home, np.nan)
    d_away = np.where(np.isfinite(facts.d_away), facts.d_away, np.nan)
    columns = {"closing_home": facts.closing[:, 0], "closing_away": facts.closing[:, 1],
               "closing_gap": facts.closing[:, 0] - facts.closing[:, 1],
               "converging_home": facts.converging[:, 0], "converging_away": facts.converging[:, 1],
               "converging_gap": facts.converging[:, 0] - facts.converging[:, 1]}
    for shift in APPROACH_SHIFTS:
        for name, values in (("home", d_home), ("away", d_away)):
            past = np.full(len(values), np.nan)
            future = np.full(len(values), np.nan)
            past[shift:] = values[shift:] - values[:-shift]
            future[:-shift] = values[shift:] - values[:-shift]
            columns["d_{}_past_{}".format(name, shift)] = past
            columns["d_{}_next_{}".format(name, shift)] = future
    for half in SHAPE_WINDOWS:
        columns["closing_gap_m{}".format(half)] = features.rolling_nanmean(columns["closing_gap"], half)
    return columns


def team_features(tracking, facts, touches=False, approach=False, contact=False, expert=False,
                  gather=False, derived_speed=False):
    columns = _shape_columns(tracking, derived_speed)
    if gather:
        columns.update(_gather_columns(tracking))
        columns.update(_clear_control_columns(facts))
    if expert:
        columns.update(_expert_columns(facts))
    if contact:
        columns.update(_contact_columns(columns, facts))
        columns.update(_reappear_columns(facts))
        columns.update(_turnaround_columns(columns))
    if touches:
        columns.update(_touch_columns(tracking, facts))
    if approach:
        columns.update(_approach_columns(facts))
    length, width = tracking.pitch
    sign = attack_sign(tracking)

    columns["ball_x"] = (facts.ball_xy[:, 0] - length / 2) * sign
    columns["ball_y"] = np.abs(facts.ball_xy[:, 1] - width / 2)
    columns["ball_speed"] = facts.ball_speed
    d_home = np.where(np.isfinite(facts.d_home), facts.d_home, np.nan)
    d_away = np.where(np.isfinite(facts.d_away), facts.d_away, np.nan)
    columns["d_home"], columns["d_away"] = d_home, d_away
    columns["d_gap"] = d_home - d_away
    columns["rel_home"], columns["rel_away"] = facts.rel_home, facts.rel_away

    control = np.where(facts.control == HOME, 1.0, np.where(facts.control == AWAY, -1.0, 0.0))
    near = np.where(facts.near == HOME, 1.0, np.where(facts.near == AWAY, -1.0, 0.0))
    for half in SIGNAL_WINDOWS:
        columns["control_{}".format(half)] = _rolling_sum(control, half)
        columns["near_{}".format(half)] = _rolling_sum(near, half)

    for name, event, values in (("ctrl", control != 0, control), ("nearest", near != 0, near),
                                ("seen", facts.ball_seen, near)):
        last, since, upcoming, until = _last_and_next(event, values)
        columns["{}_last".format(name)], columns["{}_since".format(name)] = last, since
        columns["{}_next".format(name)], columns["{}_until".format(name)] = upcoming, until
    return _matrix(columns)


def dead_features(tracking, derived_speed=False):
    """Players only. Nothing here looks at the ball."""
    speed = (features.speed_from_positions(tracking) if derived_speed
             else tracking.players[:, :, 2])
    x = tracking.players[:, :, 0] - 0.5
    y = tracking.players[:, :, 1] - 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_speed = np.nanmean(speed, 1)
        columns = {"speed": mean_speed, "count": np.isfinite(speed).sum(1).astype(float),
                   "spread_x": np.nanstd(x, 1), "spread_y": np.nanstd(y, 1),
                   "centre_x": np.abs(np.nanmean(x, 1)), "centre_y": np.abs(np.nanmean(y, 1)),
                   "fast": np.mean(np.nan_to_num(speed, nan=0.0) > FAST_SPEED, 1)}
    for half in (25, 75, 150):
        columns["speed_m{}".format(half)] = features.rolling_nanmean(mean_speed, half)
    smooth = columns["speed_m25"]
    columns["slowing"] = smooth - np.roll(smooth, SPEED_SHIFT)
    columns["rising"] = np.roll(smooth, -SPEED_SHIFT) - smooth
    columns["fast_m75"] = features.rolling_nanmean(columns["fast"], 75)
    return _matrix(columns)


class Bundle:
    """The two trained models."""

    def __init__(self, team, dead, touches=False, approach=False, contact=False, expert=False,
                 gather=False, derived_speed=False, change=None):
        self.team, self.dead, self.change = team, dead, change
        self.touches, self.approach, self.contact, self.expert = touches, approach, contact, expert
        self.gather, self.derived_speed = gather, derived_speed

    def probabilities(self, tracking, facts):
        """(P(home has the ball | in play), P(dead)) on every frame."""
        # Models saved earlier do not have every option.
        options = (getattr(self, "touches", False), getattr(self, "approach", False),
                   getattr(self, "contact", False), getattr(self, "expert", False),
                   getattr(self, "gather", False), getattr(self, "derived_speed", False))
        p_home = self.team.predict_proba(team_features(tracking, facts, *options)[0])[:, 1]
        p_dead = self.dead.predict_proba(dead_features(tracking, options[5])[0])[:, 1]
        return p_home, p_dead

    def change_probability(self, tracking, facts):
        """P(possession is changing within CHANGE_WINDOW frames), or None without a detector."""
        detector = getattr(self, "change", None)
        if detector is None:
            return None
        options = (getattr(self, "touches", False), getattr(self, "approach", False),
                   getattr(self, "contact", False), getattr(self, "expert", False),
                   getattr(self, "gather", False), getattr(self, "derived_speed", False))
        return detector.predict_proba(team_features(tracking, facts, *options)[0])[:, 1]


def fit(team_x, team_y, dead_x, dead_y, touches=False, approach=False, contact=False,
        expert=False, gather=False, derived_speed=False, change_x=None, change_y=None, seed=0):
    team = _make_model(seed).fit(team_x, team_y)
    dead = _make_model(seed).fit(dead_x, dead_y)
    change = None if change_x is None else _make_model(seed).fit(change_x, change_y)
    return Bundle(team, dead, touches, approach, contact, expert, gather, derived_speed, change)


class Ensemble:
    """Several models trained on differently degraded practice data, with their answers averaged.

    Each member sees the ball hidden, the noise drawn and the frames sampled differently, so their
    mistakes differ and partly cancel out when averaged.
    """

    def __init__(self, members):
        self.members = members

    def probabilities(self, tracking, facts):
        pairs = [member.probabilities(tracking, facts) for member in self.members]
        return (np.mean([p_home for p_home, _ in pairs], axis=0),
                np.mean([p_dead for _, p_dead in pairs], axis=0))

    def change_probability(self, tracking, facts):
        return None


def change_targets(truth, halves, window=None):
    """True on frames within window of a direct change between the two teams."""
    window = config.CHANGE_WINDOW if window is None else window
    near = np.zeros(len(truth), dtype=bool)
    for _, first, last in halves:
        block = truth[first:last + 1]
        for change in np.flatnonzero(block[1:] != block[:-1]) + 1:
            if {block[change - 1], block[change]} == {"H", "A"}:
                near[max(first, first + change - window):first + change + window] = True
    return near


def team_toll(p_change):
    """The toll for a change of team on each frame, lower where a change is likely."""
    toll = config.CHANGE_TEAM_TOLL * (1 - config.CHANGE_DISCOUNT * np.asarray(p_change))
    return np.maximum(toll, 0.25)


def votes(p_home, p_dead, out, alpha_dead=None, cue=None):
    """(T, 3) votes in log-probability units, so 'no idea' scores zero for every state."""
    alpha = config.V2_ALPHA_DEAD if alpha_dead is None else alpha_dead
    p_home = np.clip(p_home, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    p_dead = np.clip(p_dead, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    evidence = np.zeros((len(p_home), 3))
    alive = alpha * (np.log(1 - p_dead) + np.log(2.0))
    evidence[:, H] = alive + np.log(p_home) + np.log(2.0)
    evidence[:, A] = alive + np.log(1 - p_home) + np.log(2.0)
    evidence[:, D] = alpha * (np.log(p_dead) + np.log(2.0))
    evidence[out, D] += config.V2_W_OUT
    if cue is not None:
        evidence[cue, D] += config.V2_W_CUE
    return evidence


def load_bundle(path=config.MODEL_PATH):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def label_match(tracking, bundle, tags=(), restart_cue=None):
    """Labels for every frame of a match: the whole program in one call.

    tags are Smart Tagging instances; only set pieces are used. restart_cue adds the team's still-ball
    and touchline rules (config.USE_RESTART_CUE when None). Comparisons on external data switch it
    off, because the DFL ball during stoppages is reconstructed and cannot test it.
    """
    facts = features.compute(tracking)
    p_home, p_dead = bundle.probabilities(tracking, facts)
    use_cue = config.USE_RESTART_CUE if restart_cue is None else restart_cue
    cue = features.restart_cue(facts, tracking.pitch) if use_cue else None
    evidence = votes(p_home, p_dead, facts.out, cue=cue)
    p_change = bundle.change_probability(tracking, facts)
    restarts = [tag for tag in tags if tag.code == "set pieces"]
    if restarts:
        evidence = set_pieces.add_votes(evidence, restarts, config.TAG_WEIGHT_DEAD,
                                        config.TAG_LOOKBACK_S, config.TAG_WEIGHT_TEAM)
    cost = decode.transition_costs(config.V2_P_TEAM_SWITCH, config.V2_P_TO_DEAD,
                                   config.V2_P_FROM_DEAD)
    tolls = None if p_change is None else team_toll(p_change)
    return decode.decode(evidence, tracking.halves, cost, tolls)
