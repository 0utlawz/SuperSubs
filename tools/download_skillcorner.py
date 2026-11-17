"""Download the two Clasicos from SkillCorner's first open-data release.

    python tools/download_skillcorner.py <destination folder>

Automatic broadcast tracking of FC Barcelona vs Real Madrid (18 Dec 2019) and Real Madrid vs FC
Barcelona (1 Mar 2020). These matches are no longer in SkillCorner's current release, so the
download is pinned to the commit that published them. Licence at that commit: MIT, and SkillCorner
asks to be credited. https://github.com/SkillCorner/opendata

Each file is checked against the git hash GitHub publishes for it, so a rerun skips what is already
complete and a damaged download is reported instead of used.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

COMMIT = "1b99d401ddc82e366c5c20d6860878668fd5858f"
MATCHES = (2841, 3442)
USER_AGENT = "generic-client"
LISTING = "https://api.github.com/repos/SkillCorner/opendata/contents/data/matches/{}?ref={}"
CHUNK = 1 << 20


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}), timeout=60)


def git_hash(path):
    digest = hashlib.sha1("blob {}\0".format(os.path.getsize(path)).encode())
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("destination")
    args = parser.parse_args()

    failed = []
    for match in MATCHES:
        folder = os.path.join(args.destination, str(match))
        os.makedirs(folder, exist_ok=True)
        with fetch(LISTING.format(match, COMMIT)) as response:
            entries = json.load(response)
        for entry in entries:
            path = os.path.join(folder, os.path.basename(entry["name"]))
            if os.path.isfile(path) and git_hash(path) == entry["sha"]:
                print("have  {}/{}".format(match, entry["name"]), flush=True)
                continue
            with fetch(entry["download_url"]) as response, open(path + ".part", "wb") as out:
                for chunk in iter(lambda: response.read(CHUNK), b""):
                    out.write(chunk)
            os.replace(path + ".part", path)
            verified = git_hash(path) == entry["sha"]
            print("{}  {}/{}  {:.1f} MB".format("ok   " if verified else "BAD  ", match,
                                                entry["name"], os.path.getsize(path) / 1e6),
                  flush=True)
            if not verified:
                failed.append(entry["name"])

    if failed:
        sys.exit("{} files failed verification".format(len(failed)))
    print("all files verified", flush=True)


if __name__ == "__main__":
    main()
