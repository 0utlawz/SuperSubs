"""Download the external tuning data: seven Bundesliga matches published by the DFL.

    python tools/download_dfl.py <destination folder>

Dataset: Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event
data in elite soccer", Scientific Data 12:195. doi:10.1038/s41597-025-04505-y
Files:   doi:10.6084/m9.figshare.28196177
Licence: CC BY 4.0. Credit the Deutsche Fussball Liga (DFL) and cite the paper.

Every file is checked against the size and MD5 that figshare publishes, so a rerun skips what is
already complete and a damaged download is reported instead of used.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

ARTICLE = "https://api.figshare.com/v2/articles/28196177"
USER_AGENT = "generic-client"
CHUNK = 1 << 20


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}), timeout=60)


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete(path, entry):
    return (os.path.isfile(path) and os.path.getsize(path) == entry["size"]
            and md5_of(path) == entry["computed_md5"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("destination")
    args = parser.parse_args()
    os.makedirs(args.destination, exist_ok=True)

    with fetch(ARTICLE) as response:
        files = json.load(response)["files"]
    print("{} files, {:.0f} MB".format(len(files), sum(f["size"] for f in files) / 1e6), flush=True)

    failed = []
    for entry in files:
        path = os.path.join(args.destination, os.path.basename(entry["name"]))
        if complete(path, entry):
            print("have  {}".format(entry["name"]), flush=True)
            continue
        with fetch(entry["download_url"]) as response, open(path + ".part", "wb") as out:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                out.write(chunk)
        os.replace(path + ".part", path)
        if complete(path, entry):
            print("ok    {}  {:.1f} MB".format(entry["name"], entry["size"] / 1e6), flush=True)
        else:
            print("BAD   {}".format(entry["name"]), flush=True)
            failed.append(entry["name"])

    if failed:
        sys.exit("{} files failed verification".format(len(failed)))
    print("all {} files verified".format(len(files)), flush=True)


if __name__ == "__main__":
    main()
