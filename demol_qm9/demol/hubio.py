"""打包/校验 QM9 缓存子集的小工具（本地上传端与服务器下载端共用）。

只依赖标准库 + numpy：服务器上就算还没装 ``modelscope``，也已经能把下载下来的
文件按 sha256 校验一遍，确认传输没坏。``manifest.json`` 是这个约定的核心，
格式版本写在 ``MANIFEST_FORMAT`` 里。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tarfile
import time
import zipfile

import numpy as np

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = "demol-qm9-cache/v1"
CHUNK_BYTES = 1 << 20

# 一个完整缓存目录里必须存在的文件（数组名与 demol.qm9.CACHE_ARRAYS 对齐）
CACHE_JSONS = ("cache_stats.json",)


def human_size(num: float) -> str:
    """字节数 -> 人类可读（打包日志用）。"""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def sha256_file(path: str, chunk: int = CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def list_files(root: str, skip: tuple[str, ...] = (MANIFEST_NAME,)) -> list[str]:
    """目录下所有文件的相对路径（正斜杠、排序稳定），默认跳过 manifest 自己。"""
    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            if rel in skip:
                continue
            out.append(rel)
    return sorted(out)


def array_inventory(root: str) -> dict:
    """读 .npy 头部，记录每个数组的 shape/dtype（不载入数据）。"""
    inv: dict[str, dict] = {}
    for rel in list_files(root, skip=()):
        if not rel.endswith(".npy"):
            continue
        arr = np.load(os.path.join(root, rel), mmap_mode="r")
        inv[rel[: -len(".npy")]] = {
            "shape": [int(s) for s in arr.shape],
            "dtype": str(arr.dtype),
        }
        mmap = getattr(arr, "_mmap", None)
        if mmap is not None:  # Windows 上要及时放手，否则文件被锁住
            mmap.close()
    return inv


def env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass
    try:
        import modelscope  # noqa: PLC0415

        info["modelscope"] = modelscope.__version__
    except Exception:  # noqa: BLE001
        pass
    return info


def build_manifest(root: str, extra: dict | None = None) -> dict:
    """扫描目录，生成 manifest（文件清单 + sha256 + 数组清单 + 环境信息）。"""
    files = []
    for rel in list_files(root):
        path = os.path.join(root, rel)
        files.append({
            "path": rel,
            "bytes": int(os.path.getsize(path)),
            "sha256": sha256_file(path),
        })
    manifest: dict = {
        "format": MANIFEST_FORMAT,
        "kind": "qm9_cache_subset",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(files),
        "total_bytes": int(sum(f["bytes"] for f in files)),
        "files": files,
        "arrays": array_inventory(root),
        "env": env_info(),
    }
    stats_path = os.path.join(root, "cache_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as fh:
            manifest["cache_stats"] = json.load(fh)
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(root: str, manifest: dict) -> str:
    path = os.path.join(root, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return path


def read_manifest(root_or_path: str) -> dict | None:
    path = root_or_path
    if os.path.isdir(path):
        path = os.path.join(path, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_dir(root: str, manifest: dict | None = None) -> list[str]:
    """按 manifest 校验目录：缺文件、多文件、sha256 不符、数组形状不符。"""
    problems: list[str] = []
    manifest = manifest if manifest is not None else read_manifest(root)
    if manifest is None:
        return [f"{root} 下没有 {MANIFEST_NAME}，无法校验"]
    if manifest.get("format") != MANIFEST_FORMAT:
        problems.append(f"manifest 格式 {manifest.get('format')} != {MANIFEST_FORMAT}")
    expected = {f["path"]: f for f in manifest.get("files", [])}
    actual = set(list_files(root))
    for rel in sorted(set(expected) - actual):
        problems.append(f"缺文件 {rel}")
    for rel in sorted(actual - set(expected)):
        problems.append(f"多出未登记的文件 {rel}")
    for rel in sorted(actual & set(expected)):
        path = os.path.join(root, rel)
        size = int(os.path.getsize(path))
        if size != expected[rel]["bytes"]:
            problems.append(f"{rel} 大小 {size} != {expected[rel]['bytes']}")
            continue
        digest = sha256_file(path)
        if digest != expected[rel]["sha256"]:
            problems.append(f"{rel} sha256 不符（{digest[:12]}… != {expected[rel]['sha256'][:12]}…）")
    inv = array_inventory(root)
    for name, meta in manifest.get("arrays", {}).items():
        got = inv.get(name)
        if got is None:
            problems.append(f"数组 {name}.npy 不在目录里")
        elif got != meta:
            problems.append(f"数组 {name} 元信息 {got} != {meta}")
    return problems


def _safe_members(members, dst: str):
    """拒绝绝对路径与 .. 越界的归档成员。"""
    dst_abs = os.path.abspath(dst)
    for member in members:
        if isinstance(member, str):
            name = member
        else:
            name = getattr(member, "filename", None) or getattr(member, "name", None) or ""
        target = os.path.abspath(os.path.join(dst_abs, name))
        if os.path.isabs(name) or not target.startswith(dst_abs + os.sep):
            raise ValueError(f"归档成员路径越界：{name}")
        yield member


def make_archive(src_dir: str, archive_path: str, fmt: str = "tar.gz") -> dict:
    """把子集目录压成单个文件，返回可写进 manifest 的描述。

    manifest.json 会被一并打进归档（服务器解包后才能做逐文件 sha256 校验），
    所以本函数要在 ``write_manifest`` 之后调用；manifest 里记录的归档 sha256
    只写在外层那份（归档内那份没有 archive 字段，不影响校验）。
    """
    fmt = fmt.lower()
    files = list_files(src_dir, skip=())
    os.makedirs(os.path.dirname(os.path.abspath(archive_path)) or ".", exist_ok=True)
    started = time.time()
    if fmt in ("tar.gz", "tgz", "tar"):
        mode = "w:gz" if fmt in ("tar.gz", "tgz") else "w"
        with tarfile.open(archive_path, mode) as tf:
            for rel in files:
                tf.add(os.path.join(src_dir, rel), arcname=rel)
    elif fmt == "zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel in files:
                zf.write(os.path.join(src_dir, rel), arcname=rel)
    else:
        raise ValueError(f"不支持的归档格式：{fmt}（可选 tar.gz / tar / zip）")
    return {
        "path": os.path.abspath(archive_path),
        "name": os.path.basename(archive_path),
        "format": "zip" if fmt == "zip" else "tar",
        "bytes": int(os.path.getsize(archive_path)),
        "sha256": sha256_file(archive_path),
        "n_files": len(files),
        "seconds": round(time.time() - started, 2),
    }


def extract_archive(archive_path: str, dst_dir: str) -> int:
    """解开 tar/zip 归档到 dst_dir，返回文件数。"""
    os.makedirs(dst_dir, exist_ok=True)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            members = list(_safe_members(tf.getmembers(), dst_dir))
            if sys.version_info >= (3, 12):
                tf.extractall(dst_dir, members=members, filter="data")
            else:
                tf.extractall(dst_dir, members=members)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            names = [i.filename for i in zf.infolist() if not i.is_dir()]
            list(_safe_members(names, dst_dir))  # 只做越界检查
            zf.extractall(dst_dir)
    else:
        raise ValueError(f"{archive_path} 既不是 tar 也不是 zip")
    return len(list_files(dst_dir, skip=()))


def describe_manifest(manifest: dict) -> str:
    """一行摘要，日志里好看。"""
    stats = manifest.get("cache_stats", {})
    split = stats.get("split_counts") or {}
    parts = [
        f"{manifest.get('n_files', 0)} 个文件",
        human_size(manifest.get("total_bytes", 0)),
    ]
    if stats.get("n_molecules"):
        parts.append(f"{stats['n_molecules']} 个分子")
    if split:
        parts.append(f"train={split.get('train')} val={split.get('val')} test={split.get('test')}")
    return "，".join(parts)
