#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

LANG_PRIORITY = ["ja", "ko", "ru", "th"]
LANG_BCP47 = {"ja": "ja-JP", "ko": "ko-KR", "ru": "ru-RU", "th": "th-TH"}

DOMAINS = {
    "D01": "diplomacy",
    "D02": "business",
    "D03": "technology",
    "D04": "health",
    "D05": "legal",
    "D06": "education",
    "D07": "energy",
    "D08": "press",
    "D09": "culture",
    "D10": "emergency",
}

LENGTH_PLAN_PER_DOMAIN = {"S": 25, "M": 45, "L": 25, "XL": 5}
DURATION_RANGES = {"S": [3, 8], "M": [8, 15], "L": [15, 22], "XL": [22, 30]}
DIFFICULTY_COUNTS = {
    "numbers": 200,
    "proper_nouns": 150,
    "terminology": 200,
    "logic": 150,
    "enumeration": 150,
    "plain": 150,
}

UNIT_RANGE = {
    "ja": {"S": (20, 50), "M": (50, 110), "L": (110, 180), "XL": (180, 240)},
    "ko": {"S": (15, 45), "M": (45, 100), "L": (100, 170), "XL": (170, 220)},
    "ru": {"S": (8, 20), "M": (20, 45), "L": (45, 75), "XL": (75, 100)},
    "th": {"S": (15, 40), "M": (40, 90), "L": (90, 150), "XL": (150, 200)},
}

VOICE_DEFAULTS = {
    "ja": ["Kore", "Fenrir", "Aoede"],
    "ko": ["Kore", "Charon", "Puck"],
    "ru": ["Fenrir", "Enceladus", "Charon"],
    "th": ["Kore", "Puck", "Sulafat"],
}

DOMAIN_HINTS = {
    "ja": {
        "technology": "生成AI基盤",
        "business": "サプライチェーン",
        "diplomacy": "多国間協力",
        "health": "公衆衛生",
        "legal": "政策規制",
        "education": "学術連携",
        "energy": "再生可能エネルギー",
        "press": "記者会見",
        "culture": "文化交流",
        "emergency": "緊急対応",
    },
    "ko": {
        "technology": "생성형 AI 인프라",
        "business": "공급망",
        "diplomacy": "다자 협력",
        "health": "공중보건",
        "legal": "정책 규제",
        "education": "학술 협력",
        "energy": "재생에너지",
        "press": "기자회견",
        "culture": "문화 교류",
        "emergency": "긴급 대응",
    },
    "ru": {
        "technology": "инфраструктура ИИ",
        "business": "цепочка поставок",
        "diplomacy": "многостороннее сотрудничество",
        "health": "общественное здравоохранение",
        "legal": "нормативная политика",
        "education": "академическое сотрудничество",
        "energy": "возобновляемая энергия",
        "press": "пресс-конференция",
        "culture": "культурный обмен",
        "emergency": "чрезвычайное реагирование",
    },
    "th": {
        "technology": "โครงสร้างพื้นฐาน AI",
        "business": "ห่วงโซ่อุปทาน",
        "diplomacy": "ความร่วมมือพหุภาคี",
        "health": "สาธารณสุข",
        "legal": "นโยบายกำกับดูแล",
        "education": "ความร่วมมือทางวิชาการ",
        "energy": "พลังงานหมุนเวียน",
        "press": "การแถลงข่าว",
        "culture": "การแลกเปลี่ยนวัฒนธรรม",
        "emergency": "การตอบสนองฉุกเฉิน",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multilingual SI TTS corpus")
    parser.add_argument("--output-dir", default="data/corpus", help="Output corpus directory")
    parser.add_argument("--languages", nargs="+", default=LANG_PRIORITY, choices=LANG_PRIORITY)
    parser.add_argument("--per-language", type=int, default=1000, help="Target samples per language")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-url", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--offline", action="store_true", help="Disable LLM and use templates")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
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
        return len([w for w in text.strip().split(" ") if w])
    if lang == "ko":
        return len([c for c in text if c.strip()])
    return len([c for c in text if c.strip()])


def in_length_range(lang: str, tier: str, text: str) -> bool:
    lo, hi = UNIT_RANGE[lang][tier]
    units = count_units(lang, text)
    return lo <= units <= hi


def build_difficulty_pool(seed: int) -> List[str]:
    pool: List[str] = []
    for tag, count in DIFFICULTY_COUNTS.items():
        pool.extend([tag] * count)
    rnd = random.Random(seed)
    rnd.shuffle(pool)
    return pool


def make_prompt(lang: str, domain: str, tier: str, tag: str, target_units: Tuple[int, int]) -> str:
    return (
        f"Generate one {LANG_BCP47[lang]} conference-style sentence for simultaneous interpretation training. "
        f"Domain: {domain}. Length tier: {tier}. Target length range: {target_units[0]}-{target_units[1]} units. "
        f"Must include difficulty focus: {tag}. Output only the sentence in {LANG_BCP47[lang]}."
    )


def llm_generate_text(llm_url: str, api_key: str, prompt: str) -> str:
    if not requests:
        return ""
    auth_value = "Bearer " + api_key
    resp = requests.post(
        llm_url,
        headers={"Authorization": auth_value, "Content-Type": "application/json"},
        data=json.dumps({"contents": [{"parts": [{"text": prompt}]}]}),
        timeout=120,
        verify=False,
    )
    if not resp.ok:
        return ""
    payload = resp.json()
    try:
        return payload["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return ""


def fallback_text(lang: str, domain: str, tag: str, tier: str, seq: int) -> str:
    hint = DOMAIN_HINTS[lang][domain]
    number = (seq % 27) + 3
    if lang == "ja":
        return f"本会議では{hint}を中心に、第{number}四半期の進捗、予算配分、実施工程を順に報告し、必要な調整案を提示します（{tier}-{seq:03d}）。"
    if lang == "ko":
        return f"이번 회의에서는 {hint}를 중심으로 {number}분기 성과, 예산 배분, 실행 일정 순서로 보고하고 필요한 조정안을 제시하겠습니다({tier}-{seq:03d})."
    if lang == "ru":
        return f"На этом заседании по теме «{hint}» мы последовательно представим результаты за {number}-й квартал, распределение бюджета и план реализации с необходимыми корректировками ({tier}-{seq:03d})."
    return f"ในการประชุมครั้งนี้เกี่ยวกับ{hint} เราจะรายงานผลไตรมาสที่ {number} การจัดสรรงบประมาณ และแผนดำเนินงานตามลำดับ พร้อมข้อเสนอการปรับปรุงที่จำเป็น ({tier}-{seq:03d})"


def apply_difficulty(text: str, lang: str, tag: str, seq: int) -> str:
    if tag == "logic":
        prefixes = {
            "ja": "ただし、",
            "ko": "다만, ",
            "ru": "Однако ",
            "th": "อย่างไรก็ตาม ",
        }
        text = prefixes[lang] + text

    additions = {
        "ja": {
            "numbers": f" 主要指標は{seq + 10}件です。",
            "proper_nouns": " 東京国際フォーラムとASEAN事務局が参加しました。",
            "terminology": " 主要論点は推論遅延とモデル蒸留です。",
            "logic": " ただし、前提条件が満たされない場合は段階的に実施します。",
            "enumeration": " 第一に安全性、第二に効率、第三に透明性を確保します。",
            "plain": " ご清聴ありがとうございます。",
        },
        "ko": {
            "numbers": f" 핵심 지표는 {seq + 10}건입니다.",
            "proper_nouns": " 서울 코엑스와 ASEAN 사무국이 참여했습니다.",
            "terminology": " 핵심 쟁점은 추론 지연과 모델 증류입니다.",
            "logic": " 다만 전제 조건이 충족되지 않으면 단계적으로 시행하겠습니다.",
            "enumeration": " 첫째 안전성, 둘째 효율성, 셋째 투명성을 확보하겠습니다.",
            "plain": " 경청해 주셔서 감사합니다.",
        },
        "ru": {
            "numbers": f" Ключевой показатель составил {seq + 10} пунктов.",
            "proper_nouns": " В обсуждении участвовали МГИМО и секретариат АСЕАН.",
            "terminology": " Основные термины: латентность вывода и дистилляция модели.",
            "logic": " Однако при невыполнении предпосылок внедрение будет поэтапным.",
            "enumeration": " Во-первых безопасность, во-вторых эффективность, в-третьих прозрачность.",
            "plain": " Благодарю за внимание.",
        },
        "th": {
            "numbers": f" ตัวชี้วัดหลักอยู่ที่ {seq + 10} รายการ",
            "proper_nouns": " การประชุมมีตัวแทนจากศูนย์การประชุมสิริกิติ์และสำนักเลขาธิการอาเซียน",
            "terminology": " ประเด็นหลักคือความหน่วงการอนุมานและการกลั่นแบบจำลอง",
            "logic": " อย่างไรก็ตาม หากเงื่อนไขตั้งต้นไม่ครบ จะดำเนินการแบบเป็นขั้นตอน",
            "enumeration": " ประการแรกความปลอดภัย ประการที่สองประสิทธิภาพ และประการที่สามความโปร่งใส",
            "plain": " ขอบคุณทุกท่านที่รับฟัง",
        },
    }
    return (text + additions[lang][tag]).strip()


def ensure_length_range(lang: str, tier: str, text: str) -> str:
    lo, hi = UNIT_RANGE[lang][tier]
    fillers = {
        "ja": " なお、実務調整と進捗確認を継続します。",
        "ko": " 또한 실무 조정과 진행 점검을 지속하겠습니다.",
        "ru": " Дополнительно продолжим рабочую координацию и контроль сроков.",
        "th": " นอกจากนี้จะเดินหน้าการประสานงานเชิงปฏิบัติและติดตามความคืบหน้าอย่างต่อเนื่อง",
    }
    while count_units(lang, text) < lo:
        text += fillers[lang]

    if count_units(lang, text) <= hi:
        return text

    if lang == "ru":
        words = re.findall(r"\S+", text, flags=re.UNICODE)
        text = " ".join(words[:hi]).strip()
        while count_units(lang, text) > hi and " " in text:
            text = text.rsplit(" ", 1)[0].strip()
        return text

    text = "".join([c for c in text][:hi]).strip()
    while count_units(lang, text) > hi:
        text = text[:-1].strip()
    return text


def build_samples_for_language(lang: str, per_language: int, seed: int, llm_url: str, llm_api_key: str, offline: bool) -> List[dict]:
    if per_language != 1000:
        raise ValueError("Current planner expects --per-language=1000")

    rnd = random.Random(seed + hash(lang) % 997)
    difficulties = build_difficulty_pool(seed + hash(lang) % 113)
    samples: List[dict] = []

    diff_idx = 0
    for domain_code, domain in DOMAINS.items():
        seq = 1
        for tier, count in LENGTH_PLAN_PER_DOMAIN.items():
            for i in range(count):
                tag = difficulties[diff_idx]
                diff_idx += 1
                lo, hi = UNIT_RANGE[lang][tier]
                prompt = make_prompt(lang, domain, tier, tag, (lo, hi))
                text = ""
                if not offline and llm_url and llm_api_key:
                    text = llm_generate_text(llm_url, llm_api_key, prompt)
                if not text:
                    text = fallback_text(lang, domain, tag, tier, i)
                text = apply_difficulty(text, lang, tag, i)
                text = ensure_length_range(lang, tier, text)
                voice = VOICE_DEFAULTS[lang][seq % len(VOICE_DEFAULTS[lang])]
                samples.append(
                    {
                        "sample_id": f"{lang}-{domain_code}-{tier}-{seq:03d}",
                        "lang": lang,
                        "domain": domain,
                        "length_tier": tier,
                        "difficulty_tags": [tag],
                        "style": "conference_keynote",
                        "text": text,
                        "target_duration_sec": DURATION_RANGES[tier],
                        "voice": voice,
                        "group_id": None,
                    }
                )
                seq += 1

    rnd.shuffle(samples)
    samples.sort(key=lambda x: x["sample_id"])
    return samples


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    for lang in args.languages:
        rows = build_samples_for_language(
            lang=lang,
            per_language=args.per_language,
            seed=args.seed,
            llm_url=args.llm_url,
            llm_api_key=args.llm_api_key,
            offline=args.offline,
        )
        lang_path = out_dir / f"{lang}.jsonl"
        write_jsonl(lang_path, rows)
        print(f"generated {len(rows)} rows -> {lang_path}")
        all_rows.extend(rows)

    manifest_path = out_dir / "manifest.jsonl"
    all_rows.sort(key=lambda r: (LANG_PRIORITY.index(r["lang"]), r["sample_id"]))
    write_jsonl(manifest_path, all_rows)
    print(f"manifest rows={len(all_rows)} -> {manifest_path}")


if __name__ == "__main__":
    main()
