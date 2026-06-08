#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

LANG_PRIORITY = ["ja", "ko", "ru", "th"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged manifest.jsonl")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--output", default="data/corpus/manifest.jsonl")
    return parser.parse_args()


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    rows = []

    for lang in LANG_PRIORITY:
        rows.extend(read_jsonl(corpus_dir / f"{lang}.jsonl"))

    rows.sort(key=lambda r: (LANG_PRIORITY.index(r["lang"]), r["sample_id"]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"manifest rows={len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
