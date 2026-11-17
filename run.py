"""Regenerate the submission from the provided match data.

    python run.py                    # writes outputs/SuperSubs.csv

Prints no numbers derived from the match: the competition forbids giving the provided data to AI
services, and this project is developed with one. Check the file yourself with the official
validator, and never choose settings from what you see on this match.
"""

import argparse
import os

from possession import config, data, model


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="SuperSubs", help="output file stem; the official board shows it")
    args = parser.parse_args()

    tracking = data.load_tracking()
    labels = model.label_match(tracking, model.load_bundle(), data.load_tags(tracking.team_names))

    path = os.path.join(config.OUTPUT_DIR, args.name + ".csv")
    write_submission(path, labels)
    print("wrote {}".format(os.path.relpath(path, config.ROOT)))


if __name__ == "__main__":
    main()
