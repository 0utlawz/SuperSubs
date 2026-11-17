"""Download Metrica Sports' three open sample matches: tracking and events.

    python tools/download_metrica.py <destination folder>

https://github.com/metrica-sports/sample-data. No formal licence: the README asks users to be
responsible with the data and to acknowledge Metrica Sports as the source when using it publicly.
Matches 1 and 2 are in Metrica's CSV format; match 3 is in the FIFA EPTS format, the same as the
competition's tracking. The download is pinned to one commit and every file is checked against the
git hash GitHub publishes for it.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
FOLDERS = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")
USER_AGENT = "generic-client"
LISTING = "https://api.github.com/repos/metrica-sports/sample-data/contents/data/{}?ref={}"
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
    for folder_name in FOLDERS:
        folder = os.path.join(args.destination, folder_name)
        os.makedirs(folder, exist_ok=True)
        with fetch(LISTING.format(folder_name, COMMIT)) as response:
            entries = json.load(response)
        for entry in entries:
            path = os.path.join(folder, os.path.basename(entry["name"]))
            if os.path.isfile(path) and git_hash(path) == entry["sha"]:
                print("have  {}".format(entry["name"]), flush=True)
                continue
            with fetch(entry["download_url"]) as response, open(path + ".part", "wb") as out:
                for chunk in iter(lambda: response.read(CHUNK), b""):
                    out.write(chunk)
            os.replace(path + ".part", path)
            verified = git_hash(path) == entry["sha"]
            print("{}  {}  {:.1f} MB".format("ok   " if verified else "BAD  ", entry["name"],
                                             os.path.getsize(path) / 1e6), flush=True)
            if not verified:
                failed.append(entry["name"])

    if failed:
        sys.exit("{} files failed verification".format(len(failed)))
    print("all files verified", flush=True)


if __name__ == "__main__":
    main()
