"""External matches for tuning: the DFL open dataset, read into the same shape as the ATD.

Seven Bundesliga matches with official tracking and, on every frame, the ball status (in play or
not) and the team in possession. Licence CC BY 4.0: credit the Deutsche Fussball Liga (DFL) and
cite Bassek et al. (2025), doi:10.1038/s41597-025-04505-y.

This is clean, corrected tracking, which the competition data is not. hide_ball() degrades it so
that settings are chosen under conditions closer to the ones the algorithm will meet.
"""

import glob
import os
import re
import xml.etree.ElementTree as ET
from collections import namedtuple

import numpy as np

from . import config, features
from .data import HOME, AWAY, Tracking

DFL_DIR = os.environ.get("MTAH_DFL_DIR") or os.path.join(
    os.path.dirname(config.ROOT), "external-data", "dfl")

# Chosen before any result was looked at: one match from each division stays out of all tuning.
TEST_MATCHES = ("J03WN1", "J03WR9")
TRAIN_MATCHES = ("J03WMX", "J03WOH", "J03WOY", "J03WPY", "J03WQQ")
METRICA_MATCHES = ("MET1", "MET2", "MET3")
PFF_MATCHES = ("PFF10502", "PFF10512", "PFF10514", "PFF10515", "PFF3818", "PFF3827", "PFF3831",
               "PFF3839", "PFF3859")


def is_natural(match_id):
    """True for broadcast tracking that is already rough and is used as it comes, never degraded."""
    return match_id.startswith("PFF")

# Frames left empty between the halves, so no velocity is computed across half-time.
HALF_GAP = 250

KMH_PER_MS = 3.6
AIR_HEIGHT_M = 2.0  # above head height: a player's feet cannot reach it
BASE_NOISE_M = 0.3

Match = namedtuple("Match", "match_id tracking labels ball_z")

_FRAME_SET = re.compile(r'GameSection="(\w+)".*?TeamId="([^"]+)" PersonId="([^"]+)"')
_FRAME = re.compile(r'N="(\d+)".*?X="(-?[\d.]+)" Y="(-?[\d.]+)"(?: Z="(-?[\d.]+)")? D="[^"]*" '
                    r'S="([\d.]+)"')
_BALL_FLAGS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def _path(kind, match_id):
    found = glob.glob(os.path.join(DFL_DIR, "DFL_{}_*_DFL-MAT-{}.xml".format(kind, match_id)))
    if len(found) != 1:
        raise IOError("expected one {} file for {} in {}".format(kind, match_id, DFL_DIR))
    return found[0]


def _read_positions(path):
    """{(half, object): (frames, x, y, speed)} plus {half: (frames, possession, status)}."""
    tracks, flags, current = {}, {}, None
    with open(path) as handle:
        for line in handle:
            if line.startswith("<Frame "):
                match = _FRAME.search(line)
                current[0].append(int(match.group(1)))
                current[1].append(float(match.group(2)))
                current[2].append(float(match.group(3)))
                current[3].append(float(match.group(5)))
                if current[4] is not None:
                    pair = _BALL_FLAGS.search(line)
                    current[4].append(int(pair.group(1)))
                    current[5].append(int(pair.group(2)))
                    current[6].append(float(match.group(4) or 0.0))
            elif line.startswith("<FrameSet "):
                half, team, person = _FRAME_SET.search(line).groups()
                ball = team == "BALL"
                current = ([], [], [], [], [] if ball else None, [] if ball else None,
                           [] if ball else None)
                tracks[(half, team, person)] = current
                if ball:
                    flags[half] = current
    return tracks, flags


def _parse(match_id):
    info = ET.parse(_path("02_01_matchinformation", match_id)).getroot()
    pitch_el = next(info.iter("Environment"))
    pitch = (float(pitch_el.get("PitchX")), float(pitch_el.get("PitchY")))
    roles = {team.get("Role"): team for team in info.iter("Team")}
    side_of = {roles["home"].get("TeamId"): HOME, roles["guest"].get("TeamId"): AWAY}
    names = (roles["home"].get("TeamName"), roles["guest"].get("TeamName"))

    tracks, flags = _read_positions(_path("04_03_positions_raw_observed", match_id))

    people = sorted({(team, person) for _, team, person in tracks if team in side_of})
    slot_of = {who: slot for slot, who in enumerate(people)}
    team_of_slot = np.array([side_of[team] for team, _ in people], dtype=np.int8)

    offsets, halves, total = {}, [], 0
    for number, half in enumerate(("firstHalf", "secondHalf"), 1):
        frames = np.array(flags[half][0])
        offsets[half] = (int(frames[0]), total)
        halves.append((number, total, total + len(frames) - 1))
        total += len(frames) + HALF_GAP
    total -= HALF_GAP

    players = np.full((total, len(people), 3), np.nan)
    ball = np.full((total, 3), np.nan)
    ball_z = np.zeros(total)
    labels = np.full(total, "X", dtype="<U1")
    length, width = pitch

    def fill(target, values, half):
        first, base = offsets[half]
        index = np.array(values[0]) - first + base
        target[index, 0] = (np.array(values[1]) + length / 2) / length
        target[index, 1] = (np.array(values[2]) + width / 2) / width
        target[index, 2] = np.array(values[3])

    for (half, team, person), values in tracks.items():
        if team == "BALL":
            fill(ball, values, half)
            first, base = offsets[half]
            index = np.array(values[0]) - first + base
            possession, status = np.array(values[4]), np.array(values[5])
            ball_z[index] = np.array(values[6])
            labels[index] = np.where(status == 0, "D", np.where(possession == 1, "H", "A"))
        elif team in side_of:
            slot = slot_of[(team, person)]
            first, base = offsets[half]
            index = np.array(values[0]) - first + base
            keep = (index >= 0) & (index < total)
            players[index[keep], slot, 0] = (np.array(values[1])[keep] + length / 2) / length
            players[index[keep], slot, 1] = (np.array(values[2])[keep] + width / 2) / width
            players[index[keep], slot, 2] = np.array(values[3])[keep] / KMH_PER_MS

    return players, ball, team_of_slot, names, pitch, tuple(halves), labels, ball_z


def load_match(match_id):
    """One external match as a Match: tracking in the ATD's shape, and the truth on every frame.

    DFL match ids look like J03WMX; Metrica's sample matches are MET1, MET2 and MET3; PFF's World
    Cup matches are PFF followed by PFF's game id.
    """
    if match_id.startswith("MET"):
        from . import metrica
        return metrica.load_match(match_id)
    if match_id.startswith("PFF"):
        from . import pff
        return pff.load_match(match_id)
    path = os.path.join(config.CACHE_DIR, "dfl_{}.npz".format(match_id))
    if not os.path.isfile(path):
        players, ball, team_of_slot, names, pitch, halves, labels, ball_z = _parse(match_id)
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        np.savez(path, players=players, ball=ball, team_of_slot=team_of_slot, names=np.array(names),
                 pitch=np.array(pitch), halves=np.array(halves), labels=labels, ball_z=ball_z)
    cached = np.load(path)
    halves = tuple(tuple(int(v) for v in row) for row in cached["halves"])
    tracking = Tracking(cached["players"], cached["ball"], cached["team_of_slot"],
                        tuple(str(n) for n in cached["names"]),
                        tuple(float(v) for v in cached["pitch"]), halves)
    return Match(match_id, tracking, cached["labels"], cached["ball_z"])


def carry_dead_ball(match):
    """A copy of the match with the ball carried to the restart spot during every stoppage.

    The DFL files freeze the ball the moment play stops and jump it to the restart spot when play
    resumes. A real ball moves in between, and a frozen ball sitting over a line would give away
    the whole stoppage to any clue that looks at the ball. So each frozen stretch is replaced by a
    straight walk from where play stopped to where it restarted.
    """
    if match.match_id.startswith(("MET", "PFF")):
        return match  # their ball is simply not tracked in most stoppages, as in real tracking
    ball = match.tracking.ball.copy()
    for _, first, last in match.tracking.halves:
        for start, end in zip(*features.runs(match.labels[first:last + 1] == "D")):
            start, end = start + first, end + first
            if end > last:
                continue  # the half ended dead: there is no restart spot to walk to
            for axis in range(2):
                ball[start:end, axis] = np.linspace(ball[start, axis], ball[end, axis],
                                                    end - start, endpoint=False)
    return match._replace(tracking=match.tracking._replace(ball=ball))


def _smooth_noise(rng, shape, sigma, width):
    """Gaussian noise of standard deviation sigma, correlated over about width frames in time."""
    raw = rng.standard_normal(shape)
    kernel = np.ones(width) / np.sqrt(width)
    return sigma * np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 0, raw)


def hide_ball(match, seed, stoppage_bias=1.0, mean_gap_s=12.0, share=0.41, noise_m=0.3,
              player_drop=0.07, camera_half_m=None, id_swap_s=None, air_share=0.0):
    """A copy of the match degraded to look like automatic tracking.

    Only published figures are used: the ball is missing on 41% of graded frames and about 20.5 of
    22 players are tracked (DATA.md). How long the gaps last is not published, so tuning covers a
    wide spread of mean_gap_s rather than one assumed value; no measurement of the competition
    match enters here. stoppage_bias > 1 makes gaps start more often in a dead ball; settings
    should hold up with and without it. Every position also gets slowly varying noise, because an
    automatic tracker makes errors that official data does not.

    camera_half_m hides players more than that far along the pitch from where the play is, as a
    camera that follows the ball does. id_swap_s swaps the tracks of two team-mates about that
    often, because automatic tracking has no player identities and its tracks change hands.
    air_share hides that share of the ball's flights above AIR_HEIGHT_M, because the organisers
    say balls in the air are often lost.
    """
    rng = np.random.default_rng(seed)
    tracking = match.tracking
    ball = tracking.ball.copy()
    players = tracking.players.copy()
    n_frames = len(ball)
    length, width = tracking.pitch
    fps = config.FPS

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in tracking.halves:
        graded[first:last + 1] = True
    dead = match.labels == "D"
    frames = np.flatnonzero(graded)

    hidden = np.zeros(n_frames, dtype=bool)
    if air_share > 0:
        for start, end in zip(*features.runs(graded & (match.ball_z > AIR_HEIGHT_M))):
            if rng.random() < air_share:
                hidden[start:end] = True
    target = share * len(frames)
    while hidden.sum() < target:
        start = int(rng.choice(frames))
        if rng.random() > (1.0 if dead[start] else 1.0 / stoppage_bias):
            continue
        gap = max(int(rng.exponential(mean_gap_s) * fps), 1)
        hidden[start:start + gap] = True
    ball[hidden] = np.nan

    scale = np.array([length, width])
    # A bigger position error must drift more slowly, or it invents speed that real tracking does
    # not have: 0.3 m over 1 s gives the player speeds seen on real automatic tracking (Clasicos).
    drift = int(fps * max(1.0, (noise_m / BASE_NOISE_M) ** 2))
    ball[:, :2] += _smooth_noise(rng, (n_frames, 2), noise_m, drift) / scale
    players[:, :, :2] += _smooth_noise(rng, (n_frames, players.shape[1] * 2), noise_m, drift).reshape(
        n_frames, -1, 2) / scale

    if camera_half_m is not None:
        ball_x = tracking.ball[:, 0] * length
        centre = features.rolling_nanmean(ball_x, 3 * fps)
        off_camera = np.abs(players[:, :, 0] * length - centre[:, None]) > camera_half_m
        players[off_camera] = np.nan

    if id_swap_s is not None:
        for team in np.unique(tracking.team_of_slot):
            slots = np.flatnonzero(tracking.team_of_slot == team)
            for start in np.flatnonzero(rng.random(n_frames) < 1.0 / (id_swap_s * fps)):
                a, b = rng.choice(slots, 2, replace=False)
                players[start:, [a, b]] = players[start:, [b, a]]

    for slot in range(players.shape[1]):
        wanted = player_drop * n_frames
        dropped = 0
        while dropped < wanted:
            start = int(rng.integers(n_frames))
            gap = int(rng.exponential(5.0 * fps)) + 1
            players[start:start + gap, slot] = np.nan
            dropped += gap

    return match._replace(tracking=tracking._replace(players=players, ball=ball))
