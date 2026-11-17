"""Metrica Sports' three open sample matches, read into the same shape as the DFL matches.

https://github.com/metrica-sports/sample-data. No formal licence: the README asks users to be
responsible with the data and to acknowledge Metrica Sports as the source. Matches 1 and 2 are in
Metrica's CSV format, match 3 in FIFA EPTS like the competition's tracking. None has player speed,
so these matches only suit models that work speed out from positions.

The answer on every frame comes from Metrica's own events, which people recorded:

  * a team has the ball from its first pass, carry, recovery, restart or shot, until the other side
    does one of those;
  * play is dead from the moment the ball leaves the pitch, from a foul suffered, or from a shot
    that goes out or in, until the next restart;
  * challenges and balls lost change nothing on their own: a duel or a deflection is not a change
    of possession unless the other side then does something with the ball.
"""

import csv
import io
import json
import os

import numpy as np
import pandas as pd

from . import config
from .data import AWAY, HOME, Tracking, read_meta

METRICA_DIR = os.environ.get("MTAH_METRICA_DIR") or os.path.join(
    os.path.dirname(config.ROOT), "external-data", "metrica")
MATCHES = ("MET1", "MET2", "MET3")
PITCH = (105.0, 68.0)

CONTROL = ("PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT")
STOPS = ("BALL OUT", "FAULT RECEIVED")


def _folder(match_id):
    number = match_id[-1]
    return os.path.join(METRICA_DIR, "Sample_Game_" + number), "Sample_Game_" + number


def _csv_tracking(folder, stem):
    """(players (T, S, 3), ball (T, 3), team_of_slot, periods (T,)) from the two CSV files."""
    teams = []
    for side in ("Home", "Away"):
        table = pd.read_csv(os.path.join(folder, "{}_RawTrackingData_{}_Team.csv".format(stem, side)),
                            skiprows=2)
        values = table.iloc[:, 3:].to_numpy(dtype=float)
        teams.append((table["Period"].to_numpy(), values))
    periods = teams[0][0]
    home, away = teams[0][1], teams[1][1]
    ball = home[:, -2:]
    home, away = home[:, :-2], away[:, :-2]
    players = np.concatenate([home.reshape(len(home), -1, 2), away.reshape(len(away), -1, 2)], axis=1)
    team_of_slot = np.array([HOME] * (home.shape[1] // 2) + [AWAY] * (away.shape[1] // 2),
                            dtype=np.int8)
    return players, ball, team_of_slot, periods


def _epts_tracking(folder, stem):
    """The same from match 3's EPTS files, with periods from the metadata."""
    meta_path = os.path.join(folder, stem + "_metadata.xml")
    meta = read_meta(meta_path)
    n_slots = len(meta["team_of_slot"])
    with open(os.path.join(folder, stem + "_tracking.txt")) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    values = pd.read_csv(io.StringIO(text), header=None, engine="c", dtype=np.float64).to_numpy()
    width = len(meta["player_order"])
    players = values[:, 1:1 + n_slots * width].reshape(len(values), n_slots, width)
    players = players[..., [meta["player_order"].index("x"), meta["player_order"].index("y")]]
    ball = values[:, 1 + n_slots * width:]
    ball = ball[:, [meta["ball_order"].index("x"), meta["ball_order"].index("y")]]

    import xml.etree.ElementTree as ET
    params = {p.findtext("Name"): p.findtext("Value")
              for p in ET.parse(meta_path).getroot().iter("ProviderParameter")}
    periods = np.zeros(len(values), dtype=int)
    for number, name in ((1, "first_half"), (2, "second_half")):
        periods[int(params[name + "_start"]) - 1:int(params[name + "_end"])] = number
    return players, ball, meta["team_of_slot"], periods


def _csv_events(folder, stem):
    rows = []
    with open(os.path.join(folder, stem + "_RawEventsData.csv")) as handle:
        for row in csv.DictReader(handle):
            rows.append({"team": HOME if row["Team"] == "Home" else AWAY, "type": row["Type"],
                         "subtype": row["Subtype"].upper(), "period": int(row["Period"]),
                         "start": int(row["Start Frame"]), "end": int(row["End Frame"])})
    return rows


def _json_events(folder, stem, team_names):
    with open(os.path.join(folder, stem + "_events.json")) as handle:
        data = json.load(handle)["data"]
    rows = []
    for event in data:
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        rows.append({"team": HOME if event["team"]["name"] == team_names[0] else AWAY,
                     "type": event["type"]["name"],
                     "subtype": "-".join(s["name"] for s in subtypes).upper(),
                     "period": int(event["period"]), "start": int(event["start"]["frame"]),
                     "end": int(event["end"]["frame"] or event["start"]["frame"])})
    return rows


def labels_from_events(events, periods):
    """H / A / D on every frame of each period from its first kick-off, X elsewhere."""
    labels = np.full(len(periods), "X", dtype="<U1")
    halves = []
    for number in (1, 2):
        in_period = sorted((e for e in events if e["period"] == number), key=lambda e: e["start"])
        frames = np.flatnonzero(periods == number)
        kick_off = next(e for e in in_period if e["type"] == "SET PIECE")
        first, last = kick_off["start"] - 1, int(frames[-1])
        changes = []
        for event in in_period:
            team = "H" if event["team"] == HOME else "A"
            if event["type"] == "BALL OUT":
                # Recorded from the kick that sends it out; play stops when the ball leaves.
                changes.append((max(event["end"], event["start"]) - 1, "D"))
            elif event["type"] in STOPS:
                changes.append((event["start"] - 1, "D"))
            elif event["type"] == "SHOT" and ("OUT" in event["subtype"] or "GOAL" in event["subtype"]):
                changes.append((event["start"] - 1, team))
                changes.append((max(event["end"], event["start"]) - 1, "D"))
            elif event["type"] in CONTROL:
                changes.append((event["start"] - 1, team))
        changes.sort(key=lambda change: change[0])
        state = None
        for (frame, value), following in zip(changes, changes[1:] + [(last + 1, None)]):
            if frame < first:
                continue
            state = value
            labels[frame:min(following[0], last + 1)] = state
        halves.append((number, first, last))
    return labels, tuple(halves)


def load_match(match_id):
    """One sample match as a Match: tracking in the ATD's shape, and the answer on every frame."""
    from .external import Match
    cache = os.path.join(config.CACHE_DIR, "metrica_{}.npz".format(match_id))
    if not os.path.isfile(cache):
        folder, stem = _folder(match_id)
        if match_id == "MET3":
            players, ball, team_of_slot, periods = _epts_tracking(folder, stem)
            events = _json_events(folder, stem, read_meta(os.path.join(
                folder, stem + "_metadata.xml"))["team_names"])
        else:
            players, ball, team_of_slot, periods = _csv_tracking(folder, stem)
            events = _csv_events(folder, stem)
        labels, halves = labels_from_events(events, periods)
        full = np.full(players.shape[:2] + (3,), np.nan)
        full[:, :, :2] = players
        ball_full = np.full((len(ball), 3), np.nan)
        ball_full[:, :2] = ball
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        np.savez(cache, players=full, ball=ball_full, team_of_slot=team_of_slot,
                 halves=np.array(halves), labels=labels)
    cached = np.load(cache)
    halves = tuple(tuple(int(v) for v in row) for row in cached["halves"])
    tracking = Tracking(cached["players"], cached["ball"], cached["team_of_slot"],
                        ("Home", "Away"), PITCH, halves)
    return Match(match_id, tracking, cached["labels"], np.zeros(len(cached["labels"])))
