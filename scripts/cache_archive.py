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
import shutil
import sys
import tarfile
import time
from pathlib import Path

STREAM_BUFFER_BYTES = 16 * 1024 * 1024


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
            if (
                file_path.name.endswith(".tmp")
                or file_path.name == "_archive_manifest.json"
                or file_path.name.startswith(".unpacked_")
            ):
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
    """将本地缓存目录打包为单一未压缩 .tar 归档（16 MiB 大缓冲单遍流式写入）。"""
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
    for abs_path, rel_posix in entries:
        size = int(abs_path.stat().st_size)
        total_bytes += size
        manifest_files.append({"path": rel_posix, "size_bytes": size})

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

    print(
        f"[cache_archive] Packing {len(entries)} files "
        f"({total_bytes / (1024 * 1024):.1f} MiB) from {source_root} -> {target_tar}",
        flush=True,
    )
    written_bytes = 0
    with temp_tar.open("wb", buffering=STREAM_BUFFER_BYTES) as raw_out:
        with tarfile.open(fileobj=raw_out, mode="w|") as tar:
            info = tarfile.TarInfo(name="_archive_manifest.json")
            info.size = len(manifest_bytes)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(manifest_bytes))

            for idx, (abs_path, rel_posix) in enumerate(entries, start=1):
                size = manifest_files[idx - 1]["size_bytes"]
                tar.add(abs_path, arcname=rel_posix, recursive=False)
                written_bytes += size
                if idx == 1 or idx % 20 == 0 or idx == len(entries):
                    elapsed_so_far = max(time.perf_counter() - started, 1e-6)
                    speed_mib = (written_bytes / (1024 * 1024)) / elapsed_so_far
                    print(
                        f"[cache_archive] Packed {idx}/{len(entries)} files "
                        f"({written_bytes / (1024 * 1024):.1f} / {total_bytes / (1024 * 1024):.1f} MiB, "
                        f"{speed_mib:.1f} MiB/s): {rel_posix}",
                        flush=True,
                    )

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


def _marker_path(target_root: Path, modality: str) -> Path:
    clean_mod = str(modality or "eeg").strip().lower()
    return target_root / f".unpacked_{clean_mod}.json"


def _try_fast_skip_unpacked(
    source_tar: Path, target_root: Path, modality: str
) -> tuple[int, int] | None:
    """若本地已完成解包且所有文件尺寸均与清单匹配，直接秒级跳过，避免重扫云盘大文件。"""
    try:
        archive_size = int(source_tar.stat().st_size)
    except OSError:
        return None

    marker_file = _marker_path(target_root, modality)
    if marker_file.is_file():
        try:
            payload = json.loads(marker_file.read_text(encoding="utf-8"))
            if int(payload.get("archive_size_bytes", -1)) == archive_size:
                members = payload.get("members")
                if isinstance(members, list) and members:
                    total_bytes = 0
                    for item in members:
                        rel_path = str(item.get("path", ""))
                        expected_size = int(item.get("size_bytes", -1))
                        dest = _is_safe_tar_member(rel_path, target_root)
                        if not dest.is_file() or int(dest.stat().st_size) != expected_size:
                            return None
                        total_bytes += expected_size
                    return len(members), total_bytes
        except Exception:
            pass

    manifest_file = target_root / "_archive_manifest.json"
    if manifest_file.is_file():
        try:
            payload = json.loads(manifest_file.read_text(encoding="utf-8"))
            files = payload.get("files")
            if isinstance(files, list) and files:
                total_bytes = int(manifest_file.stat().st_size)
                for item in files:
                    rel_path = str(item.get("path", ""))
                    expected_size = int(item.get("size_bytes", -1))
                    dest = _is_safe_tar_member(rel_path, target_root)
                    if not dest.is_file() or int(dest.stat().st_size) != expected_size:
                        return None
                    total_bytes += expected_size
                return len(files) + 1, total_bytes
        except Exception:
            pass

    return None


def unpack_cache(
    archive: str | Path | None = None,
    cache_dir: str | Path | None = None,
    *,
    archive_dir: str | Path | None = None,
    modality: str = "eeg",
    skip_existing: bool = False,
) -> dict:
    """将云盘上的单一 .tar 归档单遍流式高速解包到本地缓存目录（零预扫描，16 MiB 顺序缓冲）。"""
    started = time.perf_counter()
    source_tar = resolve_archive_path(
        archive, archive_dir=archive_dir, modality=modality
    )
    if not source_tar.is_file():
        raise FileNotFoundError(f"未找到缓存归档文件：{source_tar}")
    target_root = Path(cache_dir).resolve() if cache_dir else default_cache_dir()
    target_root.mkdir(parents=True, exist_ok=True)

    archive_size = int(source_tar.stat().st_size)
    print(
        f"[cache_archive] Unpacking {source_tar} ({archive_size / (1024 * 1024):.1f} MiB) -> {target_root}",
        flush=True,
    )

    if skip_existing:
        fast_hit = _try_fast_skip_unpacked(source_tar, target_root, modality)
        if fast_hit is not None:
            skipped_files, total_bytes = fast_hit
            elapsed = time.perf_counter() - started
            print(
                f"[cache_archive] Fast skip: all {skipped_files} files "
                f"({total_bytes / (1024 * 1024):.1f} MiB) already present in {target_root} "
                f"({elapsed:.2f}s)",
                flush=True,
            )
            summary = {
                "action": "unpack",
                "archive": str(source_tar),
                "cache_dir": str(target_root),
                "modality": modality,
                "extracted_files": 0,
                "skipped_files": int(skipped_files),
                "total_bytes": int(total_bytes),
                "seconds": round(elapsed, 2),
            }
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            return summary

    extracted_files = 0
    skipped_files = 0
    total_bytes = 0
    idx = 0
    expected_total_files: int | None = None
    recorded_members: list[dict] = []

    with source_tar.open("rb", buffering=STREAM_BUFFER_BYTES) as raw_in:
        with tarfile.open(fileobj=raw_in, mode="r|*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                idx += 1
                member_size = int(member.size)
                dest_path = _is_safe_tar_member(member.name, target_root)
                recorded_members.append(
                    {"path": member.name, "size_bytes": member_size}
                )

                if (
                    skip_existing
                    and dest_path.is_file()
                    and int(dest_path.stat().st_size) == member_size
                ):
                    skipped_files += 1
                    total_bytes += member_size
                    if member.name == "_archive_manifest.json" and expected_total_files is None:
                        try:
                            manifest_obj = json.loads(dest_path.read_text(encoding="utf-8"))
                            if "file_count" in manifest_obj:
                                expected_total_files = int(manifest_obj["file_count"]) + 1
                        except Exception:
                            pass
                    continue

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                source_stream = tar.extractfile(member)
                if source_stream is None:
                    continue
                temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
                with temp_dest.open("wb", buffering=STREAM_BUFFER_BYTES) as out_stream:
                    shutil.copyfileobj(source_stream, out_stream, length=STREAM_BUFFER_BYTES)
                os.replace(temp_dest, dest_path)

                if member.name == "_archive_manifest.json" and expected_total_files is None:
                    try:
                        manifest_obj = json.loads(dest_path.read_text(encoding="utf-8"))
                        if "file_count" in manifest_obj:
                            expected_total_files = int(manifest_obj["file_count"]) + 1
                    except Exception:
                        pass

                extracted_files += 1
                total_bytes += member_size
                if (
                    idx == 1
                    or idx % 20 == 0
                    or (expected_total_files is not None and idx == expected_total_files)
                ):
                    elapsed_so_far = max(time.perf_counter() - started, 1e-6)
                    speed_mib = (total_bytes / (1024 * 1024)) / elapsed_so_far
                    total_str = f"/{expected_total_files}" if expected_total_files else ""
                    print(
                        f"[cache_archive] Unpacked {idx}{total_str} files "
                        f"({total_bytes / (1024 * 1024):.1f} MiB, {speed_mib:.1f} MiB/s): {member.name}",
                        flush=True,
                    )

    if recorded_members:
        try:
            marker_file = _marker_path(target_root, modality)
            marker_payload = {
                "schema_version": 1,
                "modality": modality,
                "archive": str(source_tar),
                "archive_size_bytes": archive_size,
                "unpacked_unix": time.time(),
                "members": recorded_members,
            }
            marker_file.write_text(
                json.dumps(marker_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

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
