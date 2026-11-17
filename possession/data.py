"""Read the provided tracking and Smart Tagging into numpy arrays, cached on disk.

The raw ATD is one line per video frame, "<frame_id>:<players>:<ball>", with 46 player groups and
one ball group of three numbers each. The order of the numbers inside a group is read from the
metadata XML rather than assumed, and so is the team of every track slot.
"""

import io
import os
import xml.etree.ElementTree as ET
from collections import namedtuple

import numpy as np
import pandas as pd

from . import config

HOME, AWAY = 0, 1

# players (T, S, 3) and ball (T, 3) hold x, y in 0-1 pitch units and speed in m/s. pitch is
# (length, width) in metres; halves is ((half, first frame, last frame), ...), inclusive.
Tracking = namedtuple("Tracking", "players ball team_of_slot team_names pitch halves")

# start/end are video frames, end exclusive. team is HOME, AWAY or None.
Tag = namedtuple("Tag", "code start end team kind half")


def _channel_kind(channel_id):
    """'x', 'y' or 'speed' from a channel id such as 'player10_x' or 's'."""
    tail = channel_id.rsplit("_", 1)[-1].lower()
    return {"x": "x", "y": "y", "s": "speed", "speed": "speed"}[tail]


def _ref_id(element):
    return (element.get("playerChannelId") or element.get("channelId")
            or element.get("id"))


def read_meta(meta_path):
    """Team names, team of each slot, and the channel order of player and ball groups."""
    root = ET.parse(meta_path).getroot()

    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(teams)))
    home_id = teams[0][0]  # FIFA/EPTS convention: the first team listed is the home side

    player_team = {player.get("id"): player.get("teamId") for player in root.iter("Player")}
    channel_player = {channel.get("id"): channel.get("playerId")
                      for channel in root.iter("PlayerChannel")}

    spec = root.find("DataFormatSpecifications/DataFormatSpecification")
    registers = spec.findall("SplitRegister")
    player_register = next(r for r in registers if r.find("SplitRegister") is not None)
    ball_register = next(r for r in registers if r.find("BallChannelRef") is not None)
    if registers.index(player_register) > registers.index(ball_register):
        raise ValueError("ball group precedes player groups in {}".format(meta_path))

    slots = player_register.findall("SplitRegister")
    player_order = [_channel_kind(_ref_id(ref)) for ref in slots[0].findall("PlayerChannelRef")]
    team_of_slot = []
    for slot in slots:
        refs = slot.findall("PlayerChannelRef")
        if [_channel_kind(_ref_id(ref)) for ref in refs] != player_order:
            raise ValueError("track slots disagree on channel order in {}".format(meta_path))
        team_id = player_team[channel_player[_ref_id(refs[0])]]
        team_of_slot.append(HOME if team_id == home_id else AWAY)

    ball_order = [_channel_kind(_ref_id(ref)) for ref in ball_register.findall("BallChannelRef")]
    return {
        "team_names": (teams[0][1], teams[1][1]),
        "team_of_slot": np.array(team_of_slot, dtype=np.int8),
        "player_order": player_order,
        "ball_order": ball_order,
    }


def _parse_raw(raw_path, n_slots):
    """The raw ATD as a (T, 1 + 3 * n_slots + 3) float array, one row per line."""
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, engine="c", dtype=np.float64)
    values = table.to_numpy()
    expected = 1 + 3 * n_slots + 3
    if values.shape[1] != expected:
        raise ValueError("{}: {} columns, expected {}".format(raw_path, values.shape[1], expected))
    return values


def _reorder(groups, order):
    """Put the last axis of groups into x, y, speed order."""
    return groups[..., [order.index("x"), order.index("y"), order.index("speed")]]


def load_tracking(use_cache=True):
    """The tactical ATD, one record per video frame (row i is video frame i)."""
    meta_path = config.tactical_path("_atd.xml")
    raw_path = config.tactical_path("_atd.txt")
    meta = read_meta(meta_path)

    stat = os.stat(raw_path)
    cache_path = os.path.join(config.CACHE_DIR, "tracking_{}_{}.npz".format(
        stat.st_size, int(stat.st_mtime)))
    if use_cache and os.path.isfile(cache_path):
        cached = np.load(cache_path)
        return Tracking(cached["players"], cached["ball"], meta["team_of_slot"],
                        meta["team_names"], config.PITCH, config.HALVES)

    n_slots = len(meta["team_of_slot"])
    values = _parse_raw(raw_path, n_slots)

    frame_ids = values[:, 0].astype(np.int64)
    if not np.array_equal(frame_ids, np.arange(1, config.N_FRAMES + 1)):
        raise ValueError("{}: frame ids are not exactly 1..{}".format(raw_path, config.N_FRAMES))

    players = _reorder(values[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3), meta["player_order"])
    ball = _reorder(values[:, 1 + 3 * n_slots:], meta["ball_order"])

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.savez(cache_path, players=players, ball=ball)
    return Tracking(players, ball, meta["team_of_slot"], meta["team_names"], config.PITCH,
                    config.HALVES)


def _team_of_name(name, team_names):
    """HOME / AWAY for a club name as the tagging spells it, None for 'N/A' or no tag."""
    if not name:
        return None
    wanted = name.strip().lower()
    for side, club in enumerate(team_names):
        if club and club.strip().lower() == wanted:
            return side
    return None


def load_tags(team_names):
    """Smart Tagging instances as Tags, in file order. Seconds x 25 is a video frame."""
    tags = []
    for instance in ET.parse(config.tactical_path("_smart_tagging.xml")).iter("instance"):
        labels = {label.findtext("group"): label.findtext("text")
                  for label in instance.findall("label")}
        tags.append(Tag(
            code=(instance.findtext("code") or "").strip().lower(),
            start=int(round(float(instance.findtext("start")) * config.FPS)),
            end=int(round(float(instance.findtext("end")) * config.FPS)),
            team=_team_of_name(labels.get("Team"), team_names),
            kind=(labels.get("Type") or "").strip().lower(),
            half=labels.get("Half"),
        ))
    return tags
