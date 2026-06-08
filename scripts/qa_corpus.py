#!/usr/bin/env python3
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

VALID_LANGS = ["ja", "ko", "ru", "th"]
VALID_TIERS = ["S", "M", "L", "XL"]
VALID_TAGS = ["numbers", "proper_nouns", "terminology", "logic", "enumeration", "plain"]

UNIT_RANGE = {
    "ja": {"S": (20, 50), "M": (50, 110), "L": (110, 180), "XL": (180, 240)},
    "ko": {"S": (15, 45), "M": (45, 100), "L": (100, 170), "XL": (170, 220)},
    "ru": {"S": (8, 20), "M": (20, 45), "L": (45, 75), "XL": (75, 100)},
    "th": {"S": (15, 40), "M": (40, 90), "L": (90, 150), "XL": (150, 200)},
}

TERMS = {
    "ja": ["推論", "規制", "協力", "予算", "会議"],
    "ko": ["추론", "규제", "협력", "예산", "회의"],
    "ru": ["вывода", "регулирования", "сотрудничества", "бюджета", "заседании"],
    "th": ["อนุมาน", "กำกับดูแล", "ความร่วมมือ", "งบประมาณ", "ประชุม"],
}

LOGIC_MARKERS = {
    "ja": ["ただし", "しかし", "そのため"],
    "ko": ["다만", "그러나", "따라서"],
    "ru": ["однако", "поэтому", "если"],
    "th": ["อย่างไรก็ตาม", "ดังนั้น", "หาก"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA corpus files")
    parser.add_argument("--manifest", default="data/corpus/manifest.jsonl")
    parser.add_argument("--report", default="data/corpus/qa_report.json")
    parser.add_argument("--manual", default="data/corpus/manual_review_10pct.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_units(lang: str, text: str) -> int:
    if lang == "ru":
        return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
    if lang == "th":
        return len([w for w in text.split(" ") if w])
    return len([c for c in text if c.strip()])


def check_tag_heuristic(row: dict) -> List[str]:
    lang = row.get("lang", "")
    text = row.get("text", "")
    tags = row.get("difficulty_tags", [])
    issues = []
    if not tags:
        return ["missing_difficulty_tag"]
    tag = tags[0]
    if tag == "numbers" and not re.search(r"\d", text):
        issues.append("numbers_without_digits")
    if tag == "terminology" and not any(t in text for t in TERMS.get(lang, [])):
        issues.append("terminology_not_detected")
    if tag == "logic" and not any(m in text.lower() for m in LOGIC_MARKERS.get(lang, [])):
        issues.append("logic_marker_not_detected")
    if tag == "enumeration" and not any(x in text for x in ["、", "，", ",", ";", "第一", "첫째", "Во-первых", "ประการแรก"]):
        issues.append("enumeration_marker_not_detected")
    return issues


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    rows = read_jsonl(manifest_path)

    errors = defaultdict(list)
    seen_ids = set()
    seen_text = set()
    lang_stats = Counter()
    tier_stats = Counter()
    tag_stats = Counter()

    for idx, row in enumerate(rows):
        sid = row.get("sample_id", "")
        lang = row.get("lang", "")
        tier = row.get("length_tier", "")
        text = row.get("text", "")
        tags = row.get("difficulty_tags", [])

        if not sid:
            errors["missing_sample_id"].append(idx)
        if sid in seen_ids:
            errors["duplicate_sample_id"].append(sid)
        seen_ids.add(sid)

        if not text.strip():
            errors["empty_text"].append(sid)
        if text in seen_text:
            errors["duplicate_text"].append(sid)
        seen_text.add(text)

        if lang not in VALID_LANGS:
            errors["invalid_lang"].append(sid)
        else:
            lang_stats[lang] += 1

        if tier not in VALID_TIERS:
            errors["invalid_tier"].append(sid)
        elif lang in VALID_LANGS:
            units = count_units(lang, text)
            lo, hi = UNIT_RANGE[lang][tier]
            if units < lo or units > hi:
                errors["length_out_of_range"].append({"sample_id": sid, "units": units, "range": [lo, hi]})
            tier_stats[(lang, tier)] += 1

        if not tags:
            errors["missing_tags"].append(sid)
        else:
            for tag in tags:
                if tag not in VALID_TAGS:
                    errors["invalid_tag"].append({"sample_id": sid, "tag": tag})
                tag_stats[(lang, tag)] += 1

        for issue in check_tag_heuristic(row):
            errors[issue].append(sid)

    rnd = random.Random(args.seed)
    manual_count = max(1, round(len(rows) * 0.1)) if rows else 0
    manual_rows = rnd.sample(rows, min(manual_count, len(rows))) if rows else []
    write_jsonl(Path(args.manual), manual_rows)

    report = {
        "total_rows": len(rows),
        "lang_stats": dict(lang_stats),
        "tier_stats": {f"{k[0]}:{k[1]}": v for k, v in tier_stats.items()},
        "tag_stats": {f"{k[0]}:{k[1]}": v for k, v in tag_stats.items()},
        "errors": errors,
        "manual_review_file": args.manual,
        "manual_review_count": len(manual_rows),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"qa report -> {report_path}")
    print(f"manual review list ({len(manual_rows)}) -> {args.manual}")


if __name__ == "__main__":
    main()
