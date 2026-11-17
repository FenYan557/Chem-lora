#!/usr/bin/env python
"""把打包好的 QM9 缓存（子集）上传到 ModelScope 数据集仓库。

典型流程（本地打包 -> 上传 -> 服务器下载）：

    # 本地
    python scripts/pack_subset.py --cache-dir data/qm9_cache \
                                  --out-dir data/qm9_subset_20k --size 20000
    set MODELSCOPE_API_TOKEN=xxxx          # 或 --token xxxx（魔搭网页 -> 账号设置 -> 访问令牌）
    python scripts/upload_modelscope.py --local-dir data/qm9_subset_20k \
                                        --repo-id <用户名>/demol-qm9-20k

    # 服务器
    python scripts/download_modelscope.py --repo-id <用户名>/demol-qm9-20k \
                                          --cache-dir data/qm9_cache

仓库默认建为 **私有**（``--visibility private``），下载需要同一个 token；
要公开就 ``--visibility public``。``--dry-run`` 只做本地校验、不打网络。

上传走 modelscope SDK（``HubApi.upload_folder``）；SDK 不可用时可用
``--use-cli`` 强制走 ``python -m modelscope.cli.cli upload`` 命令行。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

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


def resolve_token(args_token: str | None) -> str | None:
    """token 优先级：--token > 环境变量 MODELSCOPE_API_TOKEN。"""
    return args_token or os.environ.get(TOKEN_ENV) or None


def check_local_dir(local_dir: str) -> tuple[list[str], dict | None]:
    """本地预检：目录存在、文件齐全、manifest 自洽。返回 (问题列表, manifest)。"""
    problems: list[str] = []
    if not os.path.isdir(local_dir):
        return [f"目录不存在：{local_dir}"], None
    files = hubio.list_files(local_dir)
    if not files:
        problems.append(f"{local_dir} 是空的")
        return problems, None
    manifest = hubio.read_manifest(local_dir)
    if manifest is None:
        problems.append(
            f"{local_dir} 里没有 {hubio.MANIFEST_NAME}，服务器端无法按 sha256 校验。"
            "建议先用 scripts/pack_subset.py 打包"
        )
    else:
        problems.extend(hubio.verify_dir(local_dir, manifest))
    if not os.path.exists(os.path.join(local_dir, "cache_stats.json")):
        problems.append("没有 cache_stats.json——这看起来不像一个 DeMol 缓存目录")
    return problems, manifest


def print_plan(local_dir: str, manifest: dict | None, args) -> None:
    files = hubio.list_files(local_dir)
    total = sum(os.path.getsize(os.path.join(local_dir, f)) for f in files)
    print(f"[plan] 仓库    : {args.repo_id}（{args.repo_type}，"
          f"{'私有' if args.visibility == 'private' else '公开'}，分支 {args.revision}）")
    print(f"[plan] 目录    : {os.path.abspath(local_dir)}"
          + (f" -> 仓库内 {args.path_in_repo}/" if args.path_in_repo else ""))
    print(f"[plan] 文件    : {len(files)} 个，共 {hubio.human_size(total)}")
    if manifest is not None:
        print(f"[plan] manifest: {hubio.describe_manifest(manifest)}")
    big = sorted(files, key=lambda f: os.path.getsize(os.path.join(local_dir, f)),
                 reverse=True)[:5]
    for f in big:
        print(f"           {hubio.human_size(os.path.getsize(os.path.join(local_dir, f))):>10}  {f}")


# --------------------------------------------------------------------------
def upload_sdk(args, token: str | None) -> None:
    """modelscope SDK 上传路径。"""
    from modelscope.hub.api import HubApi  # noqa: PLC0415

    api = HubApi()
    if token:
        api.login(token)
    if args.create:
        try:
            api.create_repo(
                args.repo_id, token=token, visibility=args.visibility,
                repo_type=args.repo_type, license=args.license, exist_ok=True,
            )
            print(f"[ok  ] 仓库就绪：{args.repo_id}（{args.repo_type}，"
                  f"{'私有' if args.visibility == 'private' else '公开'}）")
        except Exception as exc:  # noqa: BLE001 - 已存在的仓库在某些版本会报错，继续上传即可
            print(f"[warn] 建仓库时出错（可能是已存在）：{exc}")
    kwargs = dict(
        repo_id=args.repo_id,
        folder_path=args.local_dir,
        repo_type=args.repo_type,
        token=token,
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message,
        allow_patterns=args.include or None,
        ignore_patterns=args.exclude or None,
        max_workers=args.max_workers,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    last_exc: Exception | None = None
    for attempt in range(1, args.retries + 2):
        try:
            print(f"[info] 上传中（第 {attempt} 次）…")
            started = time.time()
            api.upload_folder(**kwargs)
            print(f"[ok  ] 上传完成，用时 {time.time() - started:.0f}s")
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = min(30, 5 * attempt)
            print(f"[warn] 第 {attempt} 次上传失败：{exc}", file=sys.stderr)
            if attempt <= args.retries:
                print(f"       {wait}s 后重试", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"上传失败（重试 {args.retries} 次后仍不成功）：{last_exc}")


def upload_cli(args, token: str | None) -> None:
    """命令行兜底：python -m modelscope.cli.cli upload ..."""
    cmd = [sys.executable, "-m", "modelscope.cli.cli"]
    if token:
        cmd += ["--token", token]
    cmd += ["upload", args.repo_id, args.local_dir]
    if args.path_in_repo:
        cmd.append(args.path_in_repo)
    cmd += ["--repo-type", args.repo_type, "--revision", args.revision,
            "--max-workers", str(args.max_workers)]
    for pat in args.include:
        cmd += ["--include", pat]
    for pat in args.exclude:
        cmd += ["--exclude", pat]
    if args.commit_message:
        cmd += ["--commit-message", args.commit_message]
    print("[info] 走 CLI：" + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI 上传退出码 {proc.returncode}")


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--local-dir", required=True, help="pack_subset.py 打包出的目录")
    ap.add_argument("--repo-id", required=True, help="形如 用户名/仓库名，例如 alice/demol-qm9-20k")
    ap.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    ap.add_argument("--visibility", choices=("private", "public"), default="private",
                    help="仅在建仓库时生效（默认私有）")
    ap.add_argument("--license", default="apache-2.0", help="建仓库时的许可证")
    ap.add_argument("--no-create", dest="create", action="store_false",
                    help="仓库已存在，跳过 create_repo")
    ap.add_argument("--token", default=None, help=f"魔搭访问令牌（默认读环境变量 {TOKEN_ENV}）")
    ap.add_argument("--revision", default="master")
    ap.add_argument("--path-in-repo", default="", help="传到仓库内的子目录（默认根目录）")
    ap.add_argument("--include", action="append", default=[], help="只传匹配的 glob，可重复")
    ap.add_argument("--exclude", action="append", default=[], help="排除匹配的 glob，可重复")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2, help="失败重试次数（默认 2）")
    ap.add_argument("--commit-message", default=None)
    ap.add_argument("--use-cli", action="store_true", help="不用 SDK，强制走 modelscope CLI")
    ap.add_argument("--dry-run", action="store_true", help="只做本地校验并打印计划，不打网络")
    args = ap.parse_args(argv)

    problems, manifest = check_local_dir(args.local_dir)
    fatal = [p for p in problems if "manifest" not in p and "sha256" not in p]
    if fatal:
        for p in problems:
            print(f"[err ] {p}", file=sys.stderr)
        return 2
    for p in problems:  # 非致命问题（比如缺 manifest）降级为警告
        print(f"[warn] {p}", file=sys.stderr)

    print_plan(args.local_dir, manifest, args)
    if args.dry_run:
        print("[ok  ] dry-run：本地校验完成，未触网")
        return 0

    token = resolve_token(args.token)
    if token is None:
        print(f"[err ] 没有 token：传 --token，或 set {TOKEN_ENV}=xxxx",
              file=sys.stderr)
        return 2
    try:
        if args.use_cli:
            upload_cli(args, token)
        else:
            upload_sdk(args, token)
    except Exception as exc:  # noqa: BLE001
        print(f"[err ] {exc}", file=sys.stderr)
        print("       可以试 --use-cli 走命令行，或检查 token 权限（需要 write 权限）",
              file=sys.stderr)
        return 1

    print()
    print("服务器端下载（下载完会自动按 sha256 校验）：")
    print(f"  python scripts/download_modelscope.py --repo-id {args.repo_id} "
          f"--cache-dir data/qm9_cache"
          + ("" if args.visibility == "private" else "    # 公开仓库不需要 token"))
    if args.visibility == "private":
        print("  # 私有仓库：先在服务器上 export MODELSCOPE_API_TOKEN=xxxx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
