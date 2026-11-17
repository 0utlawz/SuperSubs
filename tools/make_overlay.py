"""Draw the provided tracking on the public five minutes of the tactical video, for watching.

    pip install -r tools/requirements-video.txt
    python tools/make_overlay.py                    # with the public answer key on screen
    python tools/make_overlay.py --no-key           # tracking only

Writes outputs/public_5min_tracking.mp4. That file is clipped match video: it stays on this
computer, it is gitignored, and it must never be published, uploaded or shown to an AI service.

This is for people to understand what counts as possession. It reads only what the organisers
provided and deliberately never opens our own labels: the organisers' clarification of 4 November
2026 forbids checking our output against the video.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, data  # noqa: E402

# The public window, inclusive, in video frames (DATA.md).
FIRST, LAST = 11185, 18684

COLOURS = {"H": (255, 200, 0), "A": (0, 140, 255), "D": (200, 200, 200)}  # BGR: cyan, orange, grey
NAMES = {"H": "BARCA", "A": "ATHLETIC", "D": "DEAD BALL", "X": "-"}
BALL_COLOUR = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MATRIX_COLUMNS = ["h{}{}".format(row, col) for row in range(3) for col in range(3)]


def project(matrix, x, y):
    """Normalised pitch coordinates to pixels, as in the organisers' load_homography.py."""
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return ((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator,
            (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator)


def clock(frame):
    seconds = frame / config.FPS
    return "{:d}:{:05.2f}".format(int(seconds // 60), seconds % 60)


def banner(image, lines):
    """Text lines over a dark box in the top-left corner: (text, colour) pairs."""
    height = 16 + 34 * len(lines)
    image[0:height, 0:560] = (image[0:height, 0:560] * 0.25).astype(np.uint8)
    for row, (text, colour) in enumerate(lines):
        cv2.putText(image, text, (12, 34 + 34 * row), FONT, 0.8, colour, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-key", action="store_true", help="leave the public answer key off")
    args = parser.parse_args()

    tracking = data.load_tracking()
    home = tracking.team_of_slot == data.HOME
    key = None
    if not args.no_key:
        key = pd.read_csv(os.path.join(config.OFFICIAL_REPO, "evaluation", "gt",
                                       "BAR-ATH_possession_gt_public.csv"))["label"].to_numpy()
    table = pd.read_csv(config.tactical_path("_homography.csv")).set_index("frame")
    matrices = table.loc[FIRST:LAST, MATRIX_COLUMNS].to_numpy(dtype=float).reshape(-1, 3, 3)

    capture = cv2.VideoCapture(config.tactical_path(".mp4"))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Read up to the window instead of seeking: a seek can land on the wrong frame, and every dot
    # would then sit slightly off the player it belongs to.
    for _ in range(FIRST):
        if not capture.grab():
            sys.exit("the video ended before the public window")

    path = os.path.join(config.OUTPUT_DIR, "public_5min_tracking.mp4")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), config.FPS, size)

    for frame, matrix in zip(range(FIRST, LAST + 1), matrices):
        ok, image = capture.read()
        if not ok:
            sys.exit("the video ended inside the public window")

        xs, ys = project(matrix, tracking.players[frame, :, 0], tracking.players[frame, :, 1])
        for x, y, is_home in zip(xs, ys, home):
            if np.isfinite(x) and np.isfinite(y):
                centre = (int(x), int(y))
                cv2.circle(image, centre, 11, COLOURS["H" if is_home else "A"], 2, cv2.LINE_AA)
                cv2.circle(image, centre, 2, COLOURS["H" if is_home else "A"], -1, cv2.LINE_AA)

        ball_x, ball_y = tracking.ball[frame, 0], tracking.ball[frame, 1]
        ball_seen = bool(np.isfinite(ball_x))
        if ball_seen:
            x, y = project(matrix, ball_x, ball_y)
            cv2.circle(image, (int(x), int(y)), 8, BALL_COLOUR, 3, cv2.LINE_AA)

        lines = [("video {}  frame {}".format(clock(frame), frame), (255, 255, 255))]
        if key is not None:
            lines.append(("ANSWER KEY: {}".format(NAMES[key[frame]]),
                          COLOURS.get(key[frame], (255, 255, 255))))
        lines.append(("ball tracked" if ball_seen else "BALL NOT TRACKED",
                      BALL_COLOUR if ball_seen else (0, 0, 255)))
        lines.append(("cyan = Barca   orange = Athletic", (255, 255, 255)))
        banner(image, lines)
        writer.write(image)

    writer.release()
    capture.release()

    check = cv2.VideoCapture(path)
    frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    print("wrote {}: {} frames, {:.1f} s, {:.0f} MB".format(
        os.path.relpath(path, config.ROOT), frames, frames / config.FPS,
        os.path.getsize(path) / 1e6))
    if frames != LAST - FIRST + 1:
        sys.exit("expected {} frames".format(LAST - FIRST + 1))


if __name__ == "__main__":
    main()
