import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


RATIOS = [0.10, 0.15, 0.20, 0.30, 0.50, 1.00]


def ratio_tag(ratio: float) -> str:
    return f"{int(round(ratio * 100)):03d}pct"


def read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> None:
    input_dir = args.input.resolve()
    header, rows = read_tsv(input_dir / "train_dataset.tsv")
    if "label" not in header:
        raise ValueError(f"Missing label column in {input_dir / 'train_dataset.tsv'}")
    label_idx = header.index("label")

    by_label: dict[str, list[list[str]]] = defaultdict(list)
    for row in rows:
        by_label[row[label_idx]].append(row)

    for ratio in args.ratios:
        tag = ratio_tag(ratio)
        out_dir = args.output_root / args.dataset / tag
        rng = random.Random(args.seed)
        selected = []
        print(f"[PREP] {args.dataset} {tag} -> {out_dir}")
        for label, label_rows in sorted(by_label.items()):
            shuffled = list(label_rows)
            rng.shuffle(shuffled)
            count = len(shuffled) if ratio >= 1.0 else max(1, int(round(len(shuffled) * ratio)))
            print(f"  label={label}: {count}/{len(shuffled)}")
            selected.extend(shuffled[:count])
        rng.shuffle(selected)
        write_tsv(out_dir / "train_dataset.tsv", header, selected)
        for name in ("valid_dataset.tsv", "test_dataset.tsv"):
            src = input_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ET-BERT RQ2 stratified TSV splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("rq2_data"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
