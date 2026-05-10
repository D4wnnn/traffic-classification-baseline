#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_SOURCES = [
    "ET-BERT-5x128/fine-tuning/rq2_outputs/RQ2",
    "TrafficFormer-5x128/fine-tuning/rq2_outputs/RQ2",
]


DEFAULT_SKIP_DIRNAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "wandb",
    "runs",
    "mlruns",
    "tensorboard",
    "checkpoints",
    "checkpoint",
    "weights",
    "weight",
    "ckpt",
    "saved_models",
    "saved_model",
    "snapshots",
    "artifacts",
}


DEFAULT_SKIP_EXTS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".onnx",
    ".pb",
    ".pkl",
    ".pickle",
    ".joblib",
    ".npy",
    ".npz",
}


DEFAULT_SKIP_GLOBS = {
    "*.bin-*",  # huggingface shard naming
    "*.ckpt-*",
    "*.pth.*",
    "*.pt.*",
}


@dataclass(frozen=True)
class CopyRecord:
    source: Path
    dest: Path
    size_bytes: int
    sha256: str


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _matches_any_glob(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _iter_files(root: Path, skip_dirnames: set[str]) -> Iterator[Path]:
    # Avoid following symlinks to prevent accidentally copying weight stores.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dp = Path(dirpath)

        # Prune directories in-place for efficiency.
        pruned: list[str] = []
        for d in list(dirnames):
            if d in skip_dirnames:
                dirnames.remove(d)
                pruned.append(d)
            else:
                full = dp / d
                try:
                    if full.is_symlink():
                        dirnames.remove(d)
                        pruned.append(d)
                except OSError:
                    dirnames.remove(d)
                    pruned.append(d)

        # Yield files
        for fn in filenames:
            p = dp / fn
            try:
                if p.is_symlink():
                    continue
            except OSError:
                continue
            yield p


def _should_skip_file(
    path: Path,
    *,
    skip_exts: set[str],
    skip_globs: set[str],
    max_mb: float,
) -> tuple[bool, str]:
    name = path.name
    suffix = path.suffix.lower()

    if suffix in skip_exts:
        return True, f"skip_ext:{suffix}"
    if _matches_any_glob(name, skip_globs):
        return True, "skip_glob"

    try:
        size = path.stat().st_size
    except OSError:
        return True, "stat_failed"

    if max_mb > 0 and size > int(max_mb * 1024 * 1024):
        return True, f"too_large>{max_mb}MB"

    return False, ""


def _baseline_tag_from_source(source_root: Path) -> str:
    """
    Convert a path like:
      <repo>/ET-BERT-5x128/fine-tuning/rq2_outputs/RQ2
    into:
      ET-BERT-5x128
    """
    parts = source_root.parts
    if "fine-tuning" in parts:
        return parts[parts.index("fine-tuning") - 1]
    # Fallback: last 1-2 components are likely RQ2/rq2_outputs; take parent-of-parent if so.
    if source_root.name.upper() == "RQ2":
        return source_root.parents[2].name if len(source_root.parents) >= 3 else source_root.parent.name
    return source_root.parent.name


def collect(
    repo_root: Path,
    source_relpaths: list[str],
    out_dir: Path,
    *,
    dry_run: bool,
    overwrite: bool,
    max_mb: float,
    skip_dirnames: set[str],
    skip_exts: set[str],
    skip_globs: set[str],
) -> tuple[list[CopyRecord], dict[str, int]]:
    records: list[CopyRecord] = []
    stats: dict[str, int] = {
        "copied": 0,
        "skipped": 0,
        "missing_sources": 0,
        "errors": 0,
    }

    out_dir = out_dir.resolve()
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for rel in source_relpaths:
        source_root = (repo_root / rel).resolve()
        if not source_root.exists():
            stats["missing_sources"] += 1
            print(f"[WARN] source not found: {source_root}", file=sys.stderr)
            continue
        if not source_root.is_dir():
            stats["missing_sources"] += 1
            print(f"[WARN] source is not a directory: {source_root}", file=sys.stderr)
            continue

        baseline_tag = _baseline_tag_from_source(source_root)
        for f in _iter_files(source_root, skip_dirnames):
            skip, reason = _should_skip_file(
                f, skip_exts=skip_exts, skip_globs=skip_globs, max_mb=max_mb
            )
            if skip:
                stats["skipped"] += 1
                continue

            rel_to_rq2 = f.relative_to(source_root)
            dest = out_dir / baseline_tag / rel_to_rq2

            try:
                if dest.exists() and not overwrite:
                    stats["skipped"] += 1
                    continue

                if not dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
                    size = dest.stat().st_size
                    digest = _sha256_file(dest)
                else:
                    size = f.stat().st_size
                    digest = _sha256_file(f)

                records.append(CopyRecord(source=f, dest=dest, size_bytes=size, sha256=digest))
                stats["copied"] += 1
            except Exception as e:  # noqa: BLE001
                stats["errors"] += 1
                print(f"[ERROR] failed to copy {f} -> {dest}: {e}", file=sys.stderr)

    return records, stats


def _write_manifest(out_dir: Path, records: list[CopyRecord], *, dry_run: bool) -> None:
    if dry_run:
        return
    manifest = out_dir / "MANIFEST.tsv"
    with manifest.open("w", encoding="utf-8") as w:
        w.write("sha256\tsize_bytes\tdest_relpath\tsource_abspath\n")
        for r in records:
            dest_rel = r.dest.relative_to(out_dir).as_posix()
            w.write(f"{r.sha256}\t{r.size_bytes}\t{dest_rel}\t{r.source.as_posix()}\n")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect RQ2 experiment outputs into a unified directory (excluding weights/checkpoints)."
    )
    p.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parent),
        help="Repository root (default: directory containing this script).",
    )
    p.add_argument(
        "--out-dir",
        default="collected_results/RQ2",
        help="Output directory under repo-root (default: collected_results/RQ2).",
    )
    p.add_argument(
        "--source",
        action="append",
        default=[],
        help="Relative path to an RQ2 directory to collect from. Can be provided multiple times. "
        "If omitted, uses built-in defaults for ET-BERT and TrafficFormer.",
    )
    p.add_argument(
        "--max-mb",
        type=float,
        default=200.0,
        help="Skip files larger than this size (MB). Set 0 to disable. Default: 200.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only print what would be copied.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in out-dir.",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    sources = args.source if args.source else list(DEFAULT_SOURCES)

    records, stats = collect(
        repo_root,
        sources,
        out_dir,
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        max_mb=float(args.max_mb),
        skip_dirnames=set(DEFAULT_SKIP_DIRNAMES),
        skip_exts=set(DEFAULT_SKIP_EXTS),
        skip_globs=set(DEFAULT_SKIP_GLOBS),
    )

    if args.dry_run:
        for r in records[:2000]:
            print(f"[DRY] {r.source} -> {r.dest}")
        if len(records) > 2000:
            print(f"[DRY] ... ({len(records) - 2000} more)")
    else:
        _write_manifest(out_dir, records, dry_run=False)

    print(
        "Done. "
        f"copied={stats['copied']} skipped={stats['skipped']} "
        f"missing_sources={stats['missing_sources']} errors={stats['errors']} "
        f"out_dir={out_dir}"
    )
    return 1 if stats["errors"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

