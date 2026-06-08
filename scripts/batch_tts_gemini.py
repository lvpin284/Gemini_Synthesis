#!/usr/bin/env python3
import argparse
import base64
import json
import random
import time
import wave
from pathlib import Path
from typing import Dict, Iterable, List

import requests

URL = "https://yibuapi.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"
LANG_PRIORITY = ["ja", "ko", "ru", "th"]
VOICE_CANDIDATES = {
    "ja": ["Kore", "Fenrir", "Aoede"],
    "ko": ["Kore", "Charon", "Puck"],
    "ru": ["Fenrir", "Enceladus", "Charon"],
    "th": ["Kore", "Puck", "Sulafat"],
}

STYLE_PROMPTS = {
    "ja": "Say in formal Japanese conference style, clear and moderately fast: {text}",
    "ko": "Say in formal Korean conference style, clear and moderately fast: {text}",
    "ru": "Say in formal Russian conference style, clear and moderately fast: {text}",
    "th": "Say in formal Thai conference style, clear and moderately fast: {text}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Gemini TTS synthesis")
    parser.add_argument("--manifest", default="data/corpus/manifest.jsonl")
    parser.add_argument("--output-dir", default="output/si_tts")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--verify-ssl", action="store_true", help="Enable SSL verification")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--pilot-per-lang", type=int, default=0, help="Generate only N samples per language")
    parser.add_argument("--voice", action="append", default=[], help="Override voice mapping like ja=Kore")
    parser.add_argument("--pick-voices", action="store_true", help="Interactively choose one voice per language")
    parser.add_argument("--force", action="store_true", help="Regenerate even if wav exists")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_voice_overrides(items: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        lang, voice = item.split("=", 1)
        lang = lang.strip()
        voice = voice.strip()
        if lang in VOICE_CANDIDATES:
            result[lang] = voice
    return result


def pick_voices_interactive(base: Dict[str, str]) -> Dict[str, str]:
    picked = dict(base)
    for lang in LANG_PRIORITY:
        default = picked.get(lang, VOICE_CANDIDATES[lang][0])
        print(f"[{lang}] candidates: {', '.join(VOICE_CANDIDATES[lang])}")
        value = input(f"[{lang}] choose voice (default {default}): ").strip()
        picked[lang] = value or default
    return picked


def synthesize(text: str, voice: str, api_key: str, timeout: int, verify_ssl: bool) -> bytes:
    auth_value = "Bearer " + api_key
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    resp = requests.post(
        URL,
        headers={"Authorization": auth_value, "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
        verify=verify_ssl,
    )
    resp.raise_for_status()
    data = resp.json()
    audio_b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
    return base64.b64decode(audio_b64)


def write_wav(path: Path, audio: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(audio)


def select_rows(rows: List[dict], pilot_per_lang: int) -> List[dict]:
    rows.sort(key=lambda r: (LANG_PRIORITY.index(r["lang"]), r["sample_id"]))
    if pilot_per_lang <= 0:
        return rows
    selected = []
    counts = {lang: 0 for lang in LANG_PRIORITY}
    for row in rows:
        lang = row["lang"]
        if counts[lang] >= pilot_per_lang:
            continue
        selected.append(row)
        counts[lang] += 1
    return selected


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.manifest))
    rows = select_rows(rows, args.pilot_per_lang)

    voices = {lang: VOICE_CANDIDATES[lang][0] for lang in LANG_PRIORITY}
    voices.update(parse_voice_overrides(args.voice))
    if args.pick_voices:
        voices = pick_voices_interactive(voices)

    done_path = Path(args.output_dir) / "manifest_done.jsonl"
    done_path.parent.mkdir(parents=True, exist_ok=True)

    for row in rows:
        lang = row["lang"]
        sample_id = row["sample_id"]
        voice = voices.get(lang, row.get("voice") or VOICE_CANDIDATES[lang][0])
        output_file = Path(args.output_dir) / lang / f"{sample_id}.wav"

        if output_file.exists() and not args.force:
            print(f"skip existing -> {output_file}")
            continue

        text = STYLE_PROMPTS[lang].format(text=row["text"])

        success = False
        for attempt in range(1, args.max_retries + 1):
            try:
                audio = synthesize(text, voice, args.api_key, args.timeout, args.verify_ssl)
                write_wav(output_file, audio)
                with done_path.open("a", encoding="utf-8") as f:
                    done = dict(row)
                    done["voice"] = voice
                    done["wav_path"] = str(output_file)
                    f.write(json.dumps(done, ensure_ascii=False) + "\n")
                print(f"saved -> {output_file}")
                success = True
                break
            except Exception as exc:  # pylint: disable=broad-except
                print(f"retry {attempt}/{args.max_retries} failed for {sample_id}: {exc}")
                time.sleep(random.uniform(0.5, 1.0))

        if not success:
            print(f"failed -> {sample_id}")


if __name__ == "__main__":
    main()
