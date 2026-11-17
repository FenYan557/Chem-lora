#!/usr/bin/env python
"""服务器端：从 ModelScope 拉取 QM9 缓存（子集），按 manifest 校验后变成训练可用的缓存目录。

典型用法（对应 scripts/upload_modelscope.py 的上传）：

    pip install modelscope                     # 服务器上如果没有的话
    export MODELSCOPE_API_TOKEN=xxxx           # 私有仓库需要（公开仓库可省）
    python scripts/download_modelscope.py --repo-id <用户名>/demol-qm9-20k \
                                          --cache-dir data/qm9_cache

下载完成后目录里就是标准缓存（``.npy`` + ``split.npy`` + ``cache_stats.json``），
校验通过后直接开训：

    python scripts/run_pretrain.py --cache-dir data/qm9_cache --preset base ...

下载策略：先下到 ``--download-dir``（默认 ``<cache-dir>.download``）作为暂存，
找到真正的缓存根（含 manifest.json 或 cache_stats.json 的那一层；若是 tar/zip
归档则先解包），按 manifest.json 逐文件校验 sha256，最后挪进 ``--cache-dir``。
``--skip-download`` 可以只对已下载的暂存目录做"解包 + 校验 + 归位"。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

from demol import hubio  # noqa: E402

TOKEN_ENV = "MODELSCOPE_API_TOKEN"


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------
def download_sdk(args, token: str | None) -> str:
    """SDK 下载到暂存目录，返回暂存目录路径。"""
    if args.repo_type == "dataset":
        from modelscope import dataset_snapshot_download  # noqa: PLC0415

        path = dataset_snapshot_download(
            dataset_id=args.repo_id,
            revision=args.revision,
            local_dir=args.download_dir,
            allow_file_pattern=args.include or None,
            ignore_file_pattern=args.exclude or None,
            max_workers=args.max_workers,
            token=token,
        )
    else:
        from modelscope.hub.snapshot_download import snapshot_download  # noqa: PLC0415

        path = snapshot_download(
            model_id=args.repo_id,
            revision=args.revision,
            local_dir=args.download_dir,
            allow_file_pattern=args.include or None,
            ignore_file_pattern=args.exclude or None,
            max_workers=args.max_workers,
            repo_type=args.repo_type,
            token=token,
        )
    return os.path.abspath(path)


def download_cli(args, token: str | None) -> str:
    """命令行兜底：python -m modelscope.cli.cli download ..."""
    cmd = [sys.executable, "-m", "modelscope.cli.cli"]
    if token:
        cmd += ["--token", token]
    cmd += ["download", args.repo_id,
            "--repo-type", args.repo_type,
            "--revision", args.revision,
            "--local-dir", args.download_dir,
            "--max-workers", str(args.max_workers)]
    for pat in args.include:
        cmd += ["--include", pat]
    for pat in args.exclude:
        cmd += ["--exclude", pat]
    print("[info] 走 CLI：" + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI 下载退出码 {proc.returncode}")
    return os.path.abspath(args.download_dir)


# --------------------------------------------------------------------------
# 定位缓存根 / 解包 / 归位
# --------------------------------------------------------------------------
def find_cache_root(staging: str) -> str | None:
    """在暂存目录里找缓存根：含 manifest.json 或 cache_stats.json 的最浅一层。"""
    candidates: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(staging):
        if hubio.MANIFEST_NAME in filenames or "cache_stats.json" in filenames:
            candidates.append(dirpath)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (os.path.relpath(p, staging).count(os.sep), p))
    return candidates[0]


def find_archives(staging: str) -> list[str]:
    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(staging):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if name.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
                out.append(path)
    return sorted(out, key=len)


def move_contents(src: str, dst: str) -> None:
    """把 src 里的内容挪进 dst（允许 dst 已存在；同设备用 rename，跨设备复制）。"""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.exists(d):
            if os.path.isdir(d) and not os.path.islink(d):
                shutil.rmtree(d)
            else:
                os.remove(d)
        try:
            os.replace(s, d)
        except OSError:  # 跨设备
            if os.path.isdir(s):
                shutil.copytree(s, d)
                shutil.rmtree(s)
            else:
                shutil.copy2(s, d)
                os.remove(s)


def looks_like_cache(root: str) -> bool:
    """没有 manifest 时的最低要求：cache_stats.json + 全部 .npy 都能被 numpy 读出头。"""
    if not os.path.exists(os.path.join(root, "cache_stats.json")):
        return False
    try:
        inv = hubio.array_inventory(root)
    except Exception:  # noqa: BLE001
        return False
    return bool(inv) and "pos" in inv and "targets" in inv


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo-id", required=True, help="形如 用户名/仓库名")
    ap.add_argument("--cache-dir", default=os.path.join("data", "qm9_cache"),
                    help="最终训练用的缓存目录（默认 data/qm9_cache）")
    ap.add_argument("--download-dir", default=None,
                    help="下载暂存目录（默认 <cache-dir>.download）")
    ap.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    ap.add_argument("--revision", default="master")
    ap.add_argument("--token", default=None, help=f"魔搭访问令牌（默认读环境变量 {TOKEN_ENV}）")
    ap.add_argument("--include", action="append", default=[], help="只下匹配的 glob，可重复")
    ap.add_argument("--exclude", action="append", default=[], help="排除匹配的 glob，可重复")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--skip-download", action="store_true",
                    help="暂存目录已有内容，只做解包/校验/归位")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="跳过 sha256 校验（不建议）")
    ap.add_argument("--keep-download", action="store_true",
                    help="归位后保留暂存目录（默认删掉省空间）")
    ap.add_argument("--use-cli", action="store_true", help="不用 SDK，强制走 modelscope CLI")
    args = ap.parse_args(argv)

    args.download_dir = args.download_dir or f"{args.cache_dir}.download"

    if not args.skip_download:
        token = args.token or os.environ.get(TOKEN_ENV) or None
        os.makedirs(args.download_dir, exist_ok=True)
        try:
            if args.use_cli:
                staging = download_cli(args, token)
            else:
                staging = download_sdk(args, token)
        except Exception as exc:  # noqa: BLE001
            print(f"[err ] 下载失败：{exc}", file=sys.stderr)
            print("       检查 repo-id、token（私有仓库需要）、网络；"
                  "或试 --use-cli 走命令行", file=sys.stderr)
            return 1
        print(f"[ok  ] 下载到 {staging}")
    else:
        staging = os.path.abspath(args.download_dir)
        if not os.path.isdir(staging):
            print(f"[err ] --skip-download 但暂存目录不存在：{staging}", file=sys.stderr)
            return 2

    # 1) 定位缓存根（找不到就解包归档再找一次）
    root = find_cache_root(staging)
    if root is None:
        archives = find_archives(staging)
        if not archives:
            print(f"[err ] {staging} 里既没有 manifest.json/cache_stats.json，"
                  "也没有 tar/zip 归档——下载内容可能不对", file=sys.stderr)
            return 1
        archive = archives[0]
        print(f"[info] 发现归档 {archive}，解包中…")
        hubio.extract_archive(archive, staging)
        if not args.keep_download:
            os.remove(archive)
        root = find_cache_root(staging)
        if root is None:
            print(f"[err ] 解包 {archive} 后仍找不到缓存根", file=sys.stderr)
            return 1
    print(f"[info] 缓存根：{root}")

    # 2) 校验
    manifest = hubio.read_manifest(root)
    if manifest is None:
        if looks_like_cache(root):
            print("[warn] 没有 manifest.json，只做了「能读」级别的检查（建议上传端用 "
                  "scripts/pack_subset.py 打包，带上 sha256 清单）")
        else:
            print("[err ] 既没有 manifest.json，也不像一个完整缓存目录", file=sys.stderr)
            return 1
    elif args.verify:
        problems = hubio.verify_dir(root, manifest)
        if problems:
            for p in problems:
                print(f"[err ] {p}", file=sys.stderr)
            print(f"[err ] 校验未通过（{len(problems)} 个问题），"
                  "文件在传输中损坏——删掉暂存目录后重下", file=sys.stderr)
            return 1
        print(f"[ok  ] sha256 校验通过：{hubio.describe_manifest(manifest)}")
    else:
        print("[warn] --no-verify：跳过了 sha256 校验")

    # 3) 归位
    if os.path.abspath(root) != os.path.abspath(args.cache_dir):
        move_contents(root, args.cache_dir)
        print(f"[ok  ] 已归位到 {args.cache_dir}")
    else:
        print(f"[ok  ] 缓存就在 {args.cache_dir}")
    if not args.keep_download and os.path.abspath(staging) != os.path.abspath(args.cache_dir):
        shutil.rmtree(staging, ignore_errors=True)

    stats = hubio.read_manifest(args.cache_dir)
    n_mols = (stats or {}).get("cache_stats", {}).get("n_molecules", "?")
    print()
    print(f"完成：{args.cache_dir} 里有 {n_mols} 个分子，可以直接开训：")
    print(f"  python scripts/run_pretrain.py --cache-dir {args.cache_dir} --preset base "
          "--scale cluster")
    print(f"  python scripts/run_finetune.py --cache-dir {args.cache_dir} --targets gap "
          "--scale cluster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
