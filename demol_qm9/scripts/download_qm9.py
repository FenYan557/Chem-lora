#!/usr/bin/env python
"""Download and unpack QM9 (133,885 molecules).

Sources wired up:

``figshare`` (default)  https://figshare.com/ndownloader/files/3195389
                        the ``dsgdb9nsd.xyz.tar`` archive used by every QM9
                        paper, containing one ``.xyz`` file per molecule.
``deepchem``            DeepChem's S3 mirror ``gdb9.tar.gz``.  NOTE: despite
                        the name this contains ``gdb9.sdf`` + ``gdb9.sdf.csv``
                        (not per-molecule xyz), so it must be converted with
                        ``scripts/sdf_to_xyz.py`` afterwards.
``moldata``             the same DeepChem archive re-hosted on HuggingFace
                        (``liuganghuggingface/moldata-qm9``) as
                        ``structures/qm9.tar.gz`` (53 MB); also needs
                        ``scripts/sdf_to_xyz.py``.
``moldata-mirror``      identical file via hf-mirror.com (reachable from
                        China-datacenter servers where huggingface.co and
                        figshare are blocked).

Usage::

    python scripts/download_qm9.py --out data/qm9
    python scripts/download_qm9.py --out data/qm9 --source deepchem
    python scripts/download_qm9.py --out data/qm9 --source moldata-mirror

The script is resumable: an existing archive is re-used unless ``--force``.
Network access is required, so on a sandboxed machine run it with escalated
permissions.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import urllib.request

SOURCES = {
    "figshare": (
        "https://figshare.com/ndownloader/files/3195389",
        "dsgdb9nsd.xyz.tar",
    ),
    "deepchem": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz",
        "gdb9.tar.gz",
    ),
    "moldata": (
        "https://huggingface.co/datasets/liuganghuggingface/moldata-qm9/resolve/main/structures/qm9.tar.gz",
        "qm9_sdf.tar.gz",
    ),
    "moldata-mirror": (
        "https://hf-mirror.com/datasets/liuganghuggingface/moldata-qm9/resolve/main/structures/qm9.tar.gz",
        "qm9_sdf.tar.gz",
    ),
}


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, 100.0 * downloaded / total_size)
        mb = downloaded / 1e6
        sys.stdout.write(f"\r  {mb:8.1f} MB / {total_size / 1e6:.1f} MB ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r  {downloaded / 1e6:8.1f} MB")
    sys.stdout.flush()


def download(url: str, dest: str, force: bool = False) -> str:
    if os.path.exists(dest) and not force:
        print(f"[skip] archive already present: {dest}")
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    print(f"[get ] {url}\n       -> {dest}")
    started = time.time()
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    os.replace(tmp, dest)
    print(f"\n[done] {os.path.getsize(dest) / 1e6:.1f} MB in {time.time() - started:.0f} s")
    return dest


def extract(archive: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    print(f"[xtr ] {archive} -> {out_dir}")
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(out_dir, filter="data")     # Python >= 3.12 semantics
        except TypeError:
            tar.extractall(out_dir)
    return out_dir


def find_xyz_dir(root: str) -> str | None:
    """Locate the directory that actually holds the per-molecule .xyz files."""
    for dirpath, dirnames, filenames in os.walk(root):
        xyz = [f for f in filenames if f.endswith(".xyz")]
        if len(xyz) > 1000:
            return dirpath
        del dirnames
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/qm9", help="directory to unpack into")
    ap.add_argument("--source", choices=sorted(SOURCES), default="figshare")
    ap.add_argument("--url", default=None, help="override the download URL")
    ap.add_argument("--archive", default=None, help="path/name of the archive file")
    ap.add_argument("--force", action="store_true", help="re-download an existing archive")
    ap.add_argument("--keep-archive", action="store_true", help="do not delete the tarball")
    args = ap.parse_args(argv)

    url, default_name = SOURCES[args.source]
    url = args.url or url
    archive = args.archive or os.path.join(args.out, default_name)

    download(url, archive, force=args.force)
    extract(archive, args.out)

    xyz_dir = find_xyz_dir(args.out)
    if xyz_dir is None:
        sdf = None
        csv = None
        for dirpath, _dirnames, filenames in os.walk(args.out):
            for f in filenames:
                if f.endswith(".sdf"):
                    sdf = os.path.join(dirpath, f)
                elif f.endswith(".sdf.csv"):
                    csv = os.path.join(dirpath, f)
        if sdf and csv:
            print("[info] archive holds SDF + CSV (DeepChem layout); convert it with:")
            print(f"\n    python scripts/sdf_to_xyz.py --sdf \"{sdf}\" --csv \"{csv}\" \\")
            print(f"        --out-dir {os.path.join(args.out, 'dsgdb9nsd.xyz')}\n")
            return 0
        print("[warn] no directory with >1000 .xyz files found under "
              f"{args.out}; inspect the archive layout manually.")
        return 1
    n = sum(1 for f in os.listdir(xyz_dir) if f.endswith(".xyz"))
    print(f"[ok  ] {n} molecule files in {xyz_dir}")
    if not args.keep_archive and os.path.abspath(archive).startswith(os.path.abspath(args.out)):
        os.remove(archive)
        print(f"[clean] removed {archive}")
    print("\nnext: python scripts/preprocess_qm9.py --raw-dir "
          f'"{xyz_dir}" --cache-dir data/qm9_cache')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
