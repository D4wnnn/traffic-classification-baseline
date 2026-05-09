import argparse
import csv
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


RATIOS = [0.10, 0.15, 0.20, 0.30, 0.50, 1.00]


def ratio_tag(ratio: float) -> str:
    return f"{int(round(ratio * 100)):03d}pct"


def prepare(args: argparse.Namespace) -> None:
    input_dir = args.input.resolve()
    with (input_dir / "train_dataset.tsv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    header, train_rows = rows[0], rows[1:]
    if "label" not in header:
        raise ValueError(f"Missing label column in {input_dir / 'train_dataset.tsv'}")
    label_idx = header.index("label")

    by_label: dict[str, list[list[str]]] = defaultdict(list)
    for row in train_rows:
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
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "train_dataset.tsv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            writer.writerows(selected)
        for name in ("valid_dataset.tsv", "test_dataset.tsv"):
            src = input_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)


def parse_log(args: argparse.Namespace) -> None:
    text = args.log.read_text(encoding="utf-8", errors="replace")
    blocks = text.split("Test set evaluation.")
    tail = blocks[-1] if len(blocks) > 1 else text
    metrics = {}

    patterns = {
        "accuracy": r"Accuracy:\s*([0-9.]+)",
        "macro_precision": r"Macro precision:\s*([0-9.]+)",
        "macro_recall": r"Macro recall:\s*([0-9.]+)",
        "macro_f1": r"Macro f1:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, tail)
        if match:
            metrics[key] = float(match.group(1))

    if "accuracy" not in metrics:
        match = re.search(r"Acc\. \(Correct/Total\):\s*([0-9.]+)", tail)
        if match:
            metrics["accuracy"] = float(match.group(1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[PARSE] wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TrafficFormer RQ2 split preparation and log parsing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--input", type=Path, required=True)
    prep.add_argument("--output-root", type=Path, default=Path("rq2_data"))
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(func=prepare)

    parse = subparsers.add_parser("parse-log")
    parse.add_argument("--log", type=Path, required=True)
    parse.add_argument("--output", type=Path, required=True)
    parse.set_defaults(func=parse_log)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
