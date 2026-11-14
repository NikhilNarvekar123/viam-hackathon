#!/usr/bin/env python3
"""Download a revision-pinned candidate IR; never installs packages or runs model code."""
import argparse
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

REPO = "OpenVINO/whisper-tiny.en-fp16-ov"
REVISION = "82b00e6b7ce190802f654f891257ccdbc50faadb"
LIMIT = 1_000_000_000  # Larger downloads require explicit user approval, not this tool.


def fetch(repo, revision, destination):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use owner/repository and a full immutable 40-character commit SHA")
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != revision:
        raise ValueError("Revision mismatch")
    files = [f for f in metadata["siblings"] if f["rfilename"].endswith((".json", ".txt", ".xml", ".bin"))]
    if not files or any("size" not in f for f in files):
        raise ValueError("Missing remote size metadata; refusing unbounded download")
    total = sum(f["size"] for f in files)
    if total > LIMIT:
        raise ValueError(f"{total} bytes exceeds 1 GB; ask for approval first")
    for f in files:
        name = f["rfilename"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Only flat model repositories supported")
    destination.mkdir(parents=True, exist_ok=False)
    print(f"Downloading {total} bytes to {destination}; compatibility is NOT yet validated", flush=True)
    for f in files:
        name, size = f["rfilename"], f["size"]
        target = destination / name
        partial = destination / (name + ".part")
        request_url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}"
        downloaded = 0
        with urllib.request.urlopen(request_url, timeout=120) as response, partial.open("xb") as output:
            while chunk := response.read(min(1024 * 1024, size - downloaded + 1)):
                downloaded += len(chunk)
                if downloaded > size:
                    raise ValueError(f"Remote size exceeded for {name}")
                output.write(chunk)
        if downloaded != size:
            raise ValueError(f"Incomplete download for {name}; partial file retained")
        partial.rename(target)
    (destination / "source_manifest.json").write_text(json.dumps({
        "repository": repo, "revision": revision, "bytes": total,
        "compatibility": "unvalidated candidate; run speech_smoke.py on target",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path, default=Path("models/whisper-tiny.en-fp16-ov"))
    args = parser.parse_args()
    fetch(args.repo, args.revision, args.output)


if __name__ == "__main__":
    main()
