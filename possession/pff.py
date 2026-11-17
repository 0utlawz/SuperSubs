"""PFF FC's 2022 World Cup data, read into the same shape as the DFL and Metrica matches.

Free on request from PFF FC (now Gradient Sports): fill in the form on their blog post "Get Free
Access to Our Enhanced 2022 World Cup Dataset!". Credit PFF FC as the source.

This is broadcast tracking, so it is rough in the way the competition's automatic tracking is
rough: the ball is seen on less than half of the frames, and players off the screen have estimated
positions. Unlike the DFL matches it is used as it comes, without being degraded further. Only
positions PFF marks as seen are kept for the ball; players keep their estimated positions, because
the competition's tactical camera sees nearly all of them (about 20.5 of 22).

The answer on every frame comes from PFF's own events, which people recorded:

  * a team has the ball from its first touch, pass, cross, carry, clearance, shot or restart,
    until the other side does one of those;
  * play is dead from PFF's "ball out of play" event, which they record before every restart
    (fouls and offsides included), until the restart;
  * challenges, rebounds and loose touches change nothing on their own.

Only matches whose event file has the initial touches (PFF's version of 1 May 2025 or later) are
used, so a change of team lands on the first touch, as the team's rule says, and not on the first
pass after it.
"""

import bz2
import json
import os

import numpy as np

from . import config
from .data import AWAY, HOME, Tracking

PFF_DIR = os.environ.get("MTAH_PFF_DIR") or os.path.join(
    os.path.dirname(config.ROOT), "external-data", "pff", "FIFA World Cup 2022")

# Every match with both a tracking file and an event file with initial touches, as downloaded.
MATCHES = ("PFF10502", "PFF10512", "PFF10514", "PFF10515", "PFF3818", "PFF3827", "PFF3831",
           "PFF3839", "PFF3859")

SOURCE_FPS = 29.97
HALF_GAP = 250  # empty frames between periods, as for the DFL matches
PERIODS = (1, 2, 3, 4)  # extra time counts; a penalty shoot-out does not

CONTROL = ("PA", "CR", "SH", "CL", "BC", "IT")
KICK_OFFS = ("FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF")


def _game_id(match_id):
    return match_id[len("PFF"):]


def _read_tracking(game_id):
    """(first native frame, players (N, S, 2) in metres, ball (N, 3), team_of_slot)."""
    path = os.path.join(PFF_DIR, "Tracking Data", "{}.jsonl.bz2".format(game_id))
    frames = {}
    with bz2.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("period") in PERIODS and row["frameNum"] not in frames:
                frames[row["frameNum"]] = (row.get("homePlayers") or [], row.get("awayPlayers") or [],
                                           row.get("balls") or [])
    first = min(frames)
    n = max(frames) - first + 1
    slot_of, team_of_slot = {}, []
    for home, away, _ in frames.values():
        for side, people in ((HOME, home), (AWAY, away)):
            for person in people:
                key = (side, str(person.get("jerseyNum")))
                if key not in slot_of:
                    slot_of[key] = len(team_of_slot)
                    team_of_slot.append(side)
    players = np.full((n, len(team_of_slot), 2), np.nan)
    ball = np.full((n, 3), np.nan)
    for frame, (home, away, balls) in frames.items():
        index = frame - first
        for side, people in ((HOME, home), (AWAY, away)):
            for person in people:
                if person.get("x") is not None:
                    players[index, slot_of[(side, str(person.get("jerseyNum")))]] = (
                        person["x"], person["y"])
        if balls and balls[0].get("visibility") == "VISIBLE" and balls[0].get("x") is not None:
            ball[index] = (balls[0]["x"], balls[0]["y"], balls[0].get("z") or 0.0)
    return first, players, ball, np.array(team_of_slot, dtype=np.int8)


def _read_events(game_id):
    path = os.path.join(PFF_DIR, "Event Data", "{}.json".format(game_id))
    with open(path) as handle:
        events = json.load(handle)
    return sorted(events, key=lambda e: e["startTime"])


def _changes(events, period):
    """[(video time in s, 'H' / 'A' / 'D')] for one period, in time order."""
    changes = []
    for event in events:
        game = event.get("gameEvents") or {}
        if game.get("period") != period:
            continue
        kind = game.get("gameEventType")
        possession = (event.get("possessionEvents") or {}).get("possessionEventType")
        team = "H" if game.get("homeTeam") else "A"
        if kind == "OUT":
            changes.append((event["startTime"], "D"))
        elif kind in KICK_OFFS:
            changes.append((event["startTime"], team))
        elif kind == "OTB" and (possession in CONTROL
                                or game.get("setpieceType") not in (None, "O")):
            changes.append((event["startTime"], team))
    return changes


def _bounds(events, period):
    """(kick-off time, end time) of a period, or None if it was not played."""
    start = end = None
    for event in events:
        game = event.get("gameEvents") or {}
        if game.get("period") != period:
            continue
        if game.get("gameEventType") in KICK_OFFS and start is None:
            start = event["startTime"]
        elif game.get("gameEventType") == "END":
            end = event["startTime"]
    return None if start is None or end is None else (start, end)


def _at(values, position):
    """values at fractional native positions, linear between the two nearest frames."""
    low = np.clip(np.floor(position).astype(int), 0, len(values) - 2)
    weight = (position - low).reshape((-1,) + (1,) * (values.ndim - 1))
    return (1 - weight) * values[low] + weight * values[low + 1]


def _build(match_id):
    game_id = _game_id(match_id)
    events = _read_events(game_id)
    first_frame, players_m, ball_m, team_of_slot = _read_tracking(game_id)
    stadium = next(e["stadiumMetadata"] for e in events if e.get("stadiumMetadata"))
    length, width = float(stadium["pitchLength"]), float(stadium["pitchWidth"])
    names = [None, None]
    for event in events:
        game = event.get("gameEvents") or {}
        if game.get("teamName") and game.get("homeTeam") is not None:
            names[0 if game["homeTeam"] else 1] = game["teamName"]

    blocks, halves, total = [], [], 0
    for period in PERIODS:
        bounds = _bounds(events, period)
        if bounds is None:
            continue
        times = np.arange(bounds[0], bounds[1], 1.0 / config.FPS)
        position = times * SOURCE_FPS - first_frame
        changes = _changes(events, period)
        change_times = np.array([t for t, _ in changes])
        values = np.array([v for _, v in changes])
        index = np.searchsorted(change_times, times, side="right") - 1
        labels = np.where(index >= 0, values[np.maximum(index, 0)], "D")
        blocks.append((_at(players_m, position), _at(ball_m, position), labels))
        halves.append((period, total, total + len(times) - 1))
        total += len(times) + HALF_GAP
    total -= HALF_GAP

    players = np.full((total, len(team_of_slot), 3), np.nan)
    ball = np.full((total, 3), np.nan)
    labels = np.full(total, "X", dtype="<U1")
    ball_z = np.zeros(total)
    for (period, first, last), (block_players, block_ball, block_labels) in zip(halves, blocks):
        players[first:last + 1, :, 0] = (block_players[..., 0] + length / 2) / length
        players[first:last + 1, :, 1] = (block_players[..., 1] + width / 2) / width
        ball[first:last + 1, 0] = (block_ball[:, 0] + length / 2) / length
        ball[first:last + 1, 1] = (block_ball[:, 1] + width / 2) / width
        ball_z[first:last + 1] = np.nan_to_num(block_ball[:, 2])
        labels[first:last + 1] = block_labels
    return players, ball, team_of_slot, tuple(names), (length, width), tuple(halves), labels, ball_z


def load_match(match_id):
    """One World Cup match as a Match: tracking at 25 frames a second, and the answer on every frame."""
    from .external import Match
    cache = os.path.join(config.CACHE_DIR, "pff_{}.npz".format(match_id))
    if not os.path.isfile(cache):
        players, ball, team_of_slot, names, pitch, halves, labels, ball_z = _build(match_id)
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        np.savez(cache, players=players, ball=ball, team_of_slot=team_of_slot, names=np.array(names),
                 pitch=np.array(pitch), halves=np.array(halves), labels=labels, ball_z=ball_z)
    cached = np.load(cache)
    halves = tuple(tuple(int(v) for v in row) for row in cached["halves"])
    tracking = Tracking(cached["players"], cached["ball"], cached["team_of_slot"],
                        tuple(str(n) for n in cached["names"]),
                        tuple(float(v) for v in cached["pitch"]), halves)
    return Match(match_id, tracking, cached["labels"], cached["ball_z"])
