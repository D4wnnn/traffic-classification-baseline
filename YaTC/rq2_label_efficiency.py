import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


RATIOS = [0.10, 0.15, 0.20, 0.30, 0.50, 1.00]


def ratio_tag(ratio: float) -> str:
    return f"{int(round(ratio * 100)):03d}pct"


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def prepare(args: argparse.Namespace) -> None:
    input_dir = args.input.resolve()
    train_dir = input_dir / "train"
    by_label: dict[str, list[Path]] = defaultdict(list)
    for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        by_label[class_dir.name] = sorted(p for p in class_dir.rglob("*") if p.is_file())

    for ratio in args.ratios:
        tag = ratio_tag(ratio)
        out_dir = args.output_root / args.dataset / tag
        rng = random.Random(args.seed)
        print(f"[PREP] {args.dataset} {tag} -> {out_dir}")
        for label, files in sorted(by_label.items()):
            shuffled = list(files)
            rng.shuffle(shuffled)
            count = len(shuffled) if ratio >= 1.0 else max(1, int(round(len(shuffled) * ratio)))
            print(f"  label={label}: {count}/{len(shuffled)}")
            for src in shuffled[:count]:
                link_or_copy(src, out_dir / "train" / src.relative_to(train_dir), args.copy)

        for split in ("val", "test"):
            src_split = input_dir / split
            dst_split = out_dir / split
            if src_split.exists() and not dst_split.exists():
                os.symlink(src_split, dst_split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YaTC RQ2 stratified ImageFolder splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("rq2_data"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
