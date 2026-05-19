#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read urls.txt (one URL per line) and download each file.
Sleeps 60 seconds after each attempt (success or fail).
Supports resume via .part file.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


def filename_from_url(u: str) -> str:
    name = os.path.basename(urlparse(u).path)
    return name or "track.mp3"


def download_one(session: requests.Session, url: str, out_dir: Path, referer: str, timeout: int = 120) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = filename_from_url(url)
    final_path = out_dir / fname
    part_path = out_dir / (fname + ".part")

    resume_from = part_path.stat().st_size if part_path.exists() else 0

    headers = {
        "Referer": referer,
        "Accept": "*/*",
    }
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    with session.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        mode = "ab" if ("Range" in headers and r.status_code == 206) else "wb"
        with open(part_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    part_path.replace(final_path)
    return final_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", type=Path, default=Path("urls.txt"))
    ap.add_argument("--out", type=Path, default=Path("/gardner_museum"))
    ap.add_argument("--sleep-after-file", type=float, default=3.0)
    args = ap.parse_args()

    urls = [line.strip() for line in args.urls.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(urls)} URLs from {args.urls}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
    })

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        try:
            path = download_one(session, url, args.out, referer="https://www.gardnermuseum.org/")
            print(f"  OK   saved -> {path}")
        except Exception as e:
            print(f"  FAIL {type(e).__name__}: {e}")

        print(f"  sleeping {args.sleep_after_file:.0f}s")
        time.sleep(args.sleep_after_file)


if __name__ == "__main__":
    main()