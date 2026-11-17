"""Paths, match constants, and every tunable number in the pipeline with the reason for its value.

No number below was chosen from the competition match, its video or its ground truth. The
organisers' clarification of 4 November 2026 requires every tuned parameter to be tuned on external
data, so each value comes from one of: the Laws of the Game, the team's own real-life test, or the
external DFL matches (calibrate.py, tune.py). The reason is written next to each one, so the
technical document can quote this file directly.

Units: distances in metres, speeds in metres per second, durations in frames at 25 fps unless the
name says otherwise. 
"""

import os

# -- paths ----------------------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The official challenge repository, cloned next to this one. Only needed for scoring.
OFFICIAL_REPO = os.environ.get("MTAH_REPO") or os.path.join(
    os.path.dirname(ROOT), "more-than-a-hack-2027")

# The gated match data, laid out exactly as the official repository expects it.
DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(OFFICIAL_REPO, "data")

CACHE_DIR = os.path.join(ROOT, "cache")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models")
# Version 3: three models averaged, trained on the seven DFL matches only (train.py --all).
MODEL_PATH = os.path.join(MODEL_DIR, "possession.pkl")


def candidate_path(name):
    """Where a candidate model is kept, beside the adopted one."""
    return os.path.join(MODEL_DIR, "possession_{}.pkl".format(name))

GAME = "BAR-ATH"


def tactical_path(suffix):
    """Path to one of the tactical-camera assets, e.g. tactical_path("_atd.txt")."""
    return os.path.join(DATA_DIR, "tactical", "{}_tactical{}".format(GAME, suffix))


# -- the match ------------------------------------------------------------------------------------
# Published in the official repository (games.py, DATA.md). Boundaries, not answers.

N_FRAMES = 199319
FPS = 25
PITCH = (105.3, 68.0)  # length, width in metres

# Graded video frames of each half, inclusive. Everything outside is written as X.
HALVES = ((1, 11185, 90651), (2, 110808, 190197))

# -- ball track -----------------------------------------------------------------------------------

# The ball is interpolated in a straight line across gaps up to this long. Over 1 s a ball in play
# rarely changes direction more than once, so the line is still a fair guess at where it is; beyond
# that it is invention. Interpolated frames only ever give weak evidence.
BALL_INTERP_MAX_GAP = 25

# No player runs faster than this; anything above it is a track changing hands, not a sprint.
MAX_PLAYER_SPEED = 12.0

# Velocities are centred differences over +/- this many frames (0.24 s in total). Tracking jitter
# is a few tens of centimetres per frame, which a one-frame difference would turn into several m/s
# of fake speed; three frames each side brings that down to about 1 m/s.
VELOCITY_HALF_WINDOW = 3

# Law 9: the ball is out once the whole ball is over the whole line, so its centre must be one ball
# radius past the line. The decoder needs several frames of "out" before calling a dead ball, which
# absorbs tracking error.
OUT_MARGIN = 0.11

# -- control: a player has the ball, not just near it ----------------------------------------------

# How far from a player the ball can be while they carry it, by how fast they run: the faster
# they go, the further ahead they push it. Each radius is the 90th percentile of the ball-to-
# carrier distance in that speed band on the DFL matches (calibrate.py). The 90th rather than the
# 95th because the far tail is mostly balls knocked well ahead and chased, which nobody controls.
# (upper bound of the carrier's speed in m/s, radius in m); a player standing still uses the first.
CONTROL_RADIUS_BY_SPEED = ((3.0, 1.23), (5.0, 1.47), (7.0, 1.79), (float("inf"), 3.02))

# Ball and player must move together: a pass flying past a player, or a deflection, has a large
# speed relative to them. A dribble is a series of small touches, so the ball surges ahead and
# slows down again; this is the 85th percentile of that relative speed on the DFL matches.
CONTROL_REL_SPEED = 1.73

# The controller must be clearly nearer than the nearest opponent. Inside a duel nobody controls.
CONTROL_MARGIN = 1.0

# Weaker clue: which team's player is nearest to the ball, within this distance, when nobody
# controls it. This is what the official baseline uses on its own.
NEAR_RADIUS = 2.0

# Who is coming to the ball: players within this distance, closing on it faster than this. A hard
# run is about 4 m/s or more, and 15 m is the range at which a player can still affect the ball.
APPROACH_M = 15.0
APPROACH_SPEED = 2.0

# Players around the ball: at a touch, who else is there to contest it or to receive it.
AROUND_M = 5.0

# A keeper holding the ball: inside their own box (Law 1), with the ball barely moving at their feet.
BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16
KEEPER_HOLD_M = 2.0
KEEPER_HOLD_SPEED = 3.0
KEEPER_HOLD_FRAMES = 15

# A restart being set up: the ball still for a second, or a player holding it at the touchline.
# The team's rule, checked on real in-play ball data, where neither fires. 0.5 m/s has the same 0.00%
# false alarms as 0.2 m/s and takes nothing from the competition match.
STILL_BALL_SPEED = 0.5
THROW_IN_SPEED = 1.5
TOUCHLINE_M = 1.5
RESTART_CUE_FRAMES = 25

# The team's own judgment of when possession counts (their answers of 4 November 2026). A change is
# judged this many frames after the first touch, and control means this many frames on the ball.
JUDGE_AFTER_FRAMES = 10
CONTROL_FRAMES = 15

# Control also needs no opponent close enough to take the ball: about a stride and a leg's reach.
TACKLE_M = 2.0

# -- ball state: in play or not -------------------------------------------------------------------

# A ball faster than this inside the pitch is in play: dead balls are carried, placed or rolled
# back slowly. 6.4 m/s comes from the team's own real-life throw-in test, so a short throw-in counts
# as soon as it enters the pitch. Fast dribbles also clear it, which is harmless: this vote never
# picks a side.
MOVING_SPEED = 6.4

# The whole team slows to a walk in a stoppage. Collective speed is averaged over +/- 1 s and
# compared with a fixed cut-off set on external matches: SLOW_SCALE is how far (in log speed) from
# the cut-off the clue reaches full strength.
SLOW_SMOOTH_HALF_WINDOW = 25
SLOW_CUT_SPEED = 1.8
SLOW_SCALE = 0.4
MIN_TRACKED_FOR_SPEED = 10

# -- evidence weights (frames of clean control) ----------------------------------------------------

W_CONTROL = 1.0        # a player of team k controls the ball
W_NEAR = 0.2           # team k is merely nearest to the ball
W_MOVING = 1.0         # ball travelling fast inside the pitch: in play, either team
W_OUT = 4.0            # ball seen beyond a line: dead
W_SLOW = 0.5          # collective slowness, scaled to [-1, 1]: dead (or, negative, in play)

# Smart Tagging exists only for the competition match, so its weights cannot be tuned on external
# data and the rules forbid tuning them on that match. Version 2 does not use it. These weights are
# kept only because version 1 stays in the code as a comparison.
W_TAG_PHASE = 0.08
W_TAG_SET_PIECE = 0.2  # team taking a restart, over the second half of the tagged window
W_GOAL_DEAD = 1.0      # between a goal and the kick-off that follows it

# Phase codes whose Team tag is the team WITH the ball, and those whose Team tag is the team
# WITHOUT it (the defenders, or the side that just lost it). Lower case; matched case-insensitively.
TAG_TEAM_HAS_BALL = (
    "build up", "progression", "creating chances", "attacking transition",
    "ball in final third", "ball in the box", "players in the box", "long ball",
)
TAG_TEAM_LACKS_BALL = (
    "defending in attacking third", "defending in middle third",
    "defending in defensive third", "defensive transition",
)

# -- temporal decoding ----------------------------------------------------------------------------

# Cost of each change of state, in the same units as the votes, tuned on the DFL matches (tune.py,
# tools/proof_table.py). The change of team is the team's main judgment call, because DFL's flag
# counts very brief spells while the competition says a touch without control does not count.
# 2 has the best worst case across every reading of "control" from none to 5 s, taken over normal
# and heavy tracking noise together.
P_TEAM_SWITCH = 2.0
P_TO_DEAD = 16.0
P_FROM_DEAD = 8.0

# A dead ball lasts at least this long (2 s), set by the team: a real stoppage -- ball out, whistle,
# restart -- takes longer, and on the DFL matches only 1.2% of stoppages are shorter. Without this floor, a team change could be "explained" more cheaply by
# a stoppage of a few frames (two tolls of 8) than by a change of team (one toll of 12 or more).
MIN_DEAD_FRAMES = 50

# -- version 2 ------------------------------------------------------------------------------------

# Weight of the dead-ball model, the vote for a ball seen over a line, and the tolls, all in
# log-probability units and tuned on left-out DFL matches (tune_v2.py).
V2_ALPHA_DEAD = 0.5
V2_W_OUT = 8.0
V2_W_CUE = 1.0  # a restart being set up; the DFL ball cannot test this, so it stays a modest vote
USE_RESTART_CUE = True

# The change detector: where it thinks possession is changing, the toll for a change of team falls
# by this share of itself, scaled by its probability. Tuned on the DFL tuning matches only.
CHANGE_DISCOUNT = 0.0
CHANGE_TEAM_TOLL = 4.0
CHANGE_WINDOW = 10  # frames either side of a true change that count as "changing here"
V2_P_TEAM_SWITCH = 4.0
V2_P_TO_DEAD = 12.0
V2_P_FROM_DEAD = 12.0

# Smart Tagging's set-piece tags as votes around each restart, tuned on imitation tags
# (tools/tag_test.py). The team vote added nothing measurable, so it stays at zero.
TAG_WEIGHT_DEAD = 1.0
TAG_LOOKBACK_S = 15.0
TAG_WEIGHT_TEAM = 0.0
