"""Validate a submission and score it with the official scorer, against the public five minutes.

    python score.py outputs/mtah_possession.csv

The public window is 17 possessions. It is for checking that the pipeline is correct -- frames
aligned, labels meaning what we think -- and the rules forbid using it to choose between versions
or settings. Do not tune on this number.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from possession import config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", help="frame,label CSV to check")
    args = parser.parse_args()

    evaluation = os.path.join(config.OFFICIAL_REPO, "evaluation")
    subprocess.run([sys.executable, os.path.join(evaluation, "validate_submission.py"),
                    args.submission], check=True)

    # The official scorer scores every CSV in a folder; give it a folder with just this one.
    with tempfile.TemporaryDirectory() as folder:
        shutil.copy(args.submission, folder)
        subprocess.run([sys.executable, os.path.join(evaluation, "run_evaluation.py"),
                        "--results", folder,
                        "--out", os.path.join(config.ROOT, "reports"),
                        "--no-detail"], check=True)


if __name__ == "__main__":
    main()
