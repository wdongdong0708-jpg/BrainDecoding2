"""预处理特征缓存打包与解包工具（Google Drive 单文件归档 <-> Colab 本地高速盘）。

解决 Google Drive FUSE 对成百上千个零散 `.npy` / `.json` / `.npz` 碎片文件
I/O 极慢、超时与配额限流问题：
- `pack`: 将本地高速缓存目录（如 `/content/local_cache` 或 `BRAINDECODING_CACHE_DIR`）
  打包为云盘上的单一 `.tar` 文件（如 `archives/features_eeg.tar`）。
- `unpack`: 在 GPU 训练实例启动时将云盘上的 `.tar` 秒级解包回 `/content/local_cache`。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path


MODALITY_TO_DATASET = {
    "eeg": "chineseeeg2_littleprince",
    "chineseeeg2": "chineseeeg2_littleprince",
    "chineseeeg2_littleprince": "chineseeeg2_littleprince",
    "meg": "pallier2025",
    "pallier": "pallier2025",
    "pallier2025": "pallier2025",
    "all": "all",
}


def default_cache_dir() -> Path:
    env_val = os.environ.get("BRAINDECODING_CACHE_DIR", "").strip()
    if env_val:
        return Path(env_val).resolve()
    colab_local = Path("/content/local_cache")
    if os.name != "nt" and (colab_local.exists() or Path("/content").exists()):
        return colab_local
    return (Path(__file__).resolve().parents[1] / "data" / "cache" / "derived").resolve()


def resolve_archive_path(
    archive: str | Path | None,
    *,
    archive_dir: str | Path | None = None,
    modality: str = "eeg",
) -> Path:
    if archive:
        return Path(archive).resolve()
    base_dir = Path(archive_dir) if archive_dir else Path("archives")
    clean_mod = str(modality or "eeg").strip().lower()
    return (base_dir / f"features_{clean_mod}.tar").resolve()


def _collect_cache_files(cache_dir: Path, dataset_filter: str) -> list[tuple[Path, str]]:
    """收集需要打包的 (绝对路径, 归档内相对路径) 列表。"""
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"缓存目录不存在：{cache_dir}")

    target = MODALITY_TO_DATASET.get(dataset_filter.lower(), dataset_filter)
    roots: list[Path]
    if target == "all":
        roots = [cache_dir]
    else:
        sub = cache_dir / target
        if sub.is_dir():
            roots = [sub]
        else:
            roots = [cache_dir]

    entries: list[tuple[Path, str]] = []
    for root in roots:
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.name.endswith(".tmp") or file_path.name == "_archive_manifest.json":
                continue
            rel = file_path.relative_to(cache_dir).as_posix()
            entries.append((file_path, rel))
    if not entries:
        raise FileNotFoundError(
            f"在缓存目录 {cache_dir}（筛选：{dataset_filter}）下未找到任何缓存文件。"
        )
    return entries


def pack_cache(
    cache_dir: str | Path | None = None,
    archive: str | Path | None = None,
    *,
    archive_dir: str | Path | None = None,
    modality: str = "eeg",
) -> dict:
    """将本地缓存目录打包为单一未压缩 .tar 归档（流式原子写入）。"""
    started = time.perf_counter()
    source_root = Path(cache_dir).resolve() if cache_dir else default_cache_dir()
    target_tar = resolve_archive_path(
        archive, archive_dir=archive_dir, modality=modality
    )
    entries = _collect_cache_files(source_root, modality)
    target_tar.parent.mkdir(parents=True, exist_ok=True)
    temp_tar = target_tar.with_suffix(target_tar.suffix + ".tmp")

    total_bytes = 0
    manifest_files = []
    print(
        f"[cache_archive] Packing {len(entries)} files from {source_root} -> {target_tar}",
        flush=True,
    )
    with tarfile.open(temp_tar, mode="w") as tar:
        for idx, (abs_path, rel_posix) in enumerate(entries, start=1):
            size = abs_path.stat().st_size
            total_bytes += size
            manifest_files.append({"path": rel_posix, "size_bytes": int(size)})
            tar.add(abs_path, arcname=rel_posix, recursive=False)
            if idx == 1 or idx % 20 == 0 or idx == len(entries):
                print(
                    f"[cache_archive] Packed {idx}/{len(entries)} files "
                    f"({total_bytes / (1024 * 1024):.1f} MiB): {rel_posix}",
                    flush=True,
                )

        manifest_payload = {
            "schema_version": 1,
            "modality": modality,
            "dataset": MODALITY_TO_DATASET.get(modality.lower(), modality),
            "file_count": len(entries),
            "total_bytes": int(total_bytes),
            "created_unix": time.time(),
            "files": manifest_files,
        }
        manifest_bytes = (
            json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        info = tarfile.TarInfo(name="_archive_manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(manifest_bytes))

    os.replace(temp_tar, target_tar)
    elapsed = time.perf_counter() - started
    summary = {
        "action": "pack",
        "cache_dir": str(source_root),
        "archive": str(target_tar),
        "modality": modality,
        "file_count": len(entries),
        "total_bytes": int(total_bytes),
        "archive_size_bytes": int(target_tar.stat().st_size),
        "seconds": round(elapsed, 2),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def _is_safe_tar_member(member_name: str, target_root: Path) -> Path:
    dest = (target_root / member_name).resolve()
    root_resolved = target_root.resolve()
    if dest != root_resolved and root_resolved not in dest.parents:
        raise ValueError(f"检测到越界 tar 路径条目：{member_name}")
    return dest


def unpack_cache(
    archive: str | Path | None = None,
    cache_dir: str | Path | None = None,
    *,
    archive_dir: str | Path | None = None,
    modality: str = "eeg",
    skip_existing: bool = False,
) -> dict:
    """将云盘上的单一 .tar 归档高速解包到本地缓存目录。"""
    started = time.perf_counter()
    source_tar = resolve_archive_path(
        archive, archive_dir=archive_dir, modality=modality
    )
    if not source_tar.is_file():
        raise FileNotFoundError(f"未找到缓存归档文件：{source_tar}")
    target_root = Path(cache_dir).resolve() if cache_dir else default_cache_dir()
    target_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[cache_archive] Unpacking {source_tar} -> {target_root}",
        flush=True,
    )
    extracted_files = 0
    skipped_files = 0
    total_bytes = 0

    with tarfile.open(source_tar, mode="r:*") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        total_members = len(members)
        for idx, member in enumerate(members, start=1):
            dest_path = _is_safe_tar_member(member.name, target_root)
            if (
                skip_existing
                and dest_path.is_file()
                and dest_path.stat().st_size == member.size
            ):
                skipped_files += 1
                total_bytes += int(member.size)
                continue
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            source_stream = tar.extractfile(member)
            if source_stream is None:
                continue
            temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
            with temp_dest.open("wb") as out_stream:
                while True:
                    chunk = source_stream.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    out_stream.write(chunk)
            os.replace(temp_dest, dest_path)
            extracted_files += 1
            total_bytes += int(member.size)
            if idx == 1 or idx % 20 == 0 or idx == total_members:
                print(
                    f"[cache_archive] Unpacked {idx}/{total_members} files "
                    f"({total_bytes / (1024 * 1024):.1f} MiB): {member.name}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    summary = {
        "action": "unpack",
        "archive": str(source_tar),
        "cache_dir": str(target_root),
        "modality": modality,
        "extracted_files": extracted_files,
        "skipped_files": skipped_files,
        "total_bytes": int(total_bytes),
        "seconds": round(elapsed, 2),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="预处理特征缓存打包与解包工具（pack / unpack）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_p = subparsers.add_parser(
        "pack", help="将本地缓存目录打包为单一 .tar 文件"
    )
    pack_p.add_argument(
        "--cache-dir",
        default=None,
        help="本地高速缓存目录（默认读取 BRAINDECODING_CACHE_DIR 或 /content/local_cache）",
    )
    pack_p.add_argument(
        "--archive",
        default=None,
        help="输出 .tar 归档完整路径（如 /content/drive/MyDrive/BrainDecoding/archives/features_eeg.tar）",
    )
    pack_p.add_argument(
        "--archive-dir",
        default=None,
        help="归档所在目录（未传 --archive 时生成 <archive-dir>/features_<modality>.tar）",
    )
    pack_p.add_argument(
        "--modality",
        default="eeg",
        help="模态或数据集标识（eeg / meg / chineseeeg2_littleprince / pallier2025 / all）",
    )

    unpack_p = subparsers.add_parser(
        "unpack", help="将单一 .tar 归档解包回本地高速缓存目录"
    )
    unpack_p.add_argument(
        "--archive",
        default=None,
        help="输入 .tar 归档完整路径",
    )
    unpack_p.add_argument(
        "--archive-dir",
        default=None,
        help="归档所在目录（未传 --archive 时读取 <archive-dir>/features_<modality>.tar）",
    )
    unpack_p.add_argument(
        "--cache-dir",
        default=None,
        help="目标本地高速缓存目录（默认读取 BRAINDECODING_CACHE_DIR 或 /content/local_cache）",
    )
    unpack_p.add_argument(
        "--modality",
        default="eeg",
        help="模态或数据集标识（eeg / meg / chineseeeg2_littleprince / pallier2025 / all）",
    )
    unpack_p.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标文件已存在且大小一致则跳过解压",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pack":
        pack_cache(
            cache_dir=args.cache_dir,
            archive=args.archive,
            archive_dir=args.archive_dir,
            modality=args.modality,
        )
        return 0
    if args.command == "unpack":
        unpack_cache(
            archive=args.archive,
            cache_dir=args.cache_dir,
            archive_dir=args.archive_dir,
            modality=args.modality,
            skip_existing=args.skip_existing,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
