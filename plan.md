# 同传多语种 TTS 数据生产计划

## 1. 背景与目标

首批聚焦日语（ja）、韩语（ko）、俄语（ru）、泰语（th）四语，每语 1000 条，共 4000 条会议口播风格 TTS 音频，用于同传与语音翻译系统评测。

## 2. 规模与分布

- 语种优先级：**ja → ko → ru → th**
- 每语种：1000 条
- 句长配比：S/M/L/XL = 250/450/250/50
- 领域配比：10 领域，每领域 100 条
- 难点标签配比：
  - numbers: 200
  - proper_nouns: 150
  - terminology: 200
  - logic: 150
  - enumeration: 150
  - plain: 150

## 3. 目录与产物

- `data/schema.json`：样本字段规范
- `data/corpus/{ja,ko,ru,th}.jsonl`：四语语料
- `data/corpus/manifest.jsonl`：合成任务清单
- `scripts/generate_corpus.py`：按配比分语种生成文本
- `scripts/qa_corpus.py`：自动质检 + 10% 人工抽检清单
- `scripts/build_manifest.py`：合并四语到 manifest
- `scripts/batch_tts_gemini.py`：批量合成（断点续传/重试/按语种目录）

## 4. 生产流程

1. 先生成四语文本语料
2. 跑自动质检与人工抽检清单
3. 先每语 25 条（共 100 条）试点合成
4. 验收后再全量合成 4000 条

## 5. TTS 约束

- API：`gemini-2.5-flash-preview-tts:generateContent`
- 输出：24kHz、16-bit、单声道 WAV
- 网络：支持 `verify=False`
- 容错：单条失败重试 3 次，支持断点续传
