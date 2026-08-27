"""Download and pin the content-corpus raw data (ticket 02, MATS arc).

    uv run python scripts/fetch_content_data.py [--congress 114]

Two sources, both kept under data/raw/ with sha256s recorded in a
provenance.json per source directory:

- Gentzkow, Shapiro, Taddy, "Congressional Record for the 43rd-114th
  Congresses: Parsed Speeches and Phrase Counts" (Stanford SDR,
  purl.stanford.edu/md374tz9962), daily edition. License: ODC-BY 1.0.
  The 2.8 GB archive is downloaded once; only the one requested congress's
  speeches/descr/SpeakerMap files are extracted, and the zip itself is kept
  as the pin.
- Voteview member ideology (HSall_members.csv, voteview.com/data), the
  DW-NOMINATE scores. No formal license; citation requested (Lewis et al.,
  Voteview: Congressional Roll-Call Votes Database).

Already-present files are not re-downloaded, so re-running is cheap and a
partially-fetched state repairs itself.
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import date
from urllib.request import urlretrieve

from polreps.config import DATA_RAW

HEIN_URL = "https://stacks.stanford.edu/file/druid:md374tz9962/hein-daily.zip"
VOTEVIEW_URL = "https://voteview.com/static/data/out/members/HSall_members.csv"

HEIN_CITATION = (
    "Gentzkow, Matthew, Jesse M. Shapiro, and Matt Taddy. Congressional "
    "Record for the 43rd-114th Congresses: Parsed Speeches and Phrase "
    "Counts. Stanford Libraries [distributor], 2018-01-16. "
    "https://data.stanford.edu/congress_text"
)
VOTEVIEW_CITATION = (
    "Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron "
    "Rudkin, and Luke Sonnet. Voteview: Congressional Roll-Call Votes "
    "Database. https://voteview.com/"
)


def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"using existing {dest.name}")
        return
    print(f"downloading {url} ...")
    partial = dest.with_suffix(dest.suffix + ".part")
    urlretrieve(url, partial)
    partial.rename(dest)


def extract_congress_files(zip_path, congress, out_dir):
    """Pull the three per-congress files out of the hein-daily archive,
    matching by basename so the archive's internal folder layout is free to
    vary."""
    wanted = {
        f"speeches_{congress:03d}.txt",
        f"descr_{congress:03d}.txt",
        f"{congress:03d}_SpeakerMap.txt",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            basename = info.filename.rsplit("/", 1)[-1]
            if basename in wanted:
                if basename in found:
                    raise ValueError(f"{basename} appears twice in the archive")
                found[basename] = info
        missing = wanted - set(found)
        if missing:
            raise ValueError(f"archive lacks {sorted(missing)}")
        for basename, info in sorted(found.items()):
            dest = out_dir / basename
            if dest.exists():
                print(f"using existing {dest.name}")
                continue
            with archive.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"extracted {basename} ({dest.stat().st_size / 1e6:.0f} MB)")
    return sorted(wanted)


def write_provenance(directory, record):
    path = directory / "provenance.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--congress", type=int, default=114,
                        help="daily edition covers 97-114; 114 is the most recent")
    args = parser.parse_args()

    hein_dir = DATA_RAW / "hein-daily"
    zip_path = DATA_RAW / "downloads" / "hein-daily.zip"
    download(HEIN_URL, zip_path)
    extracted = extract_congress_files(zip_path, args.congress, hein_dir)
    write_provenance(
        hein_dir,
        {
            "source_url": HEIN_URL,
            "retrieved": date.today().isoformat(),
            "license": "ODC-BY 1.0 (https://opendatacommons.org/licenses/by/1-0/)",
            "citation": HEIN_CITATION,
            "archive_sha256": sha256_of(zip_path),
            "congress": args.congress,
            "files": {name: sha256_of(hein_dir / name) for name in extracted},
        },
    )

    voteview_dir = DATA_RAW / "voteview"
    members_csv = voteview_dir / "HSall_members.csv"
    download(VOTEVIEW_URL, members_csv)
    write_provenance(
        voteview_dir,
        {
            "source_url": VOTEVIEW_URL,
            "retrieved": date.today().isoformat(),
            "license": "no formal license stated; citation requested",
            "citation": VOTEVIEW_CITATION,
            "files": {"HSall_members.csv": sha256_of(members_csv)},
        },
    )


if __name__ == "__main__":
    main()
