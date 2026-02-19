<h1 align="center">YASTS — Yet Another Subtitle Translator Script</h1>

YASTS (Yet Another Subtitle Translator Script) is a standalone CLI tool that translates `.srt` subtitle files locally using a `llama.cpp` server.

It’s designed first and foremost to keep subtitles **structurally correct** (timings, cue order, and cue boundaries) while still giving the model enough **context** to produce consistent translations — and it’s also **fast**.

**Primary target model:** [TranslateGemma](https://huggingface.co/collections/google/translategemma) (tested)  
**Also works well with:** Llama 3–based instruct models (tested)  
**Might work with:** Qwen instruct models (not tested)

---

## Why this exists

If you’ve tried subtitle translation with local LLMs, you’ve probably seen some of these:

- translations improve with context, but large prompts cause truncation or failures
- models merge/split subtitle segments and you lose alignment with the original SRT structure
- splitting a sentence across cues makes the model “reconstruct” content across boundaries
- JSON output occasionally breaks (e.g., unescaped quotes inside strings)
- line splitting produces awkward 3-line cues even when 2 lines would fit

YASTS is built around sentence-aware grouping, windowed translation, and robust post-processing so that the output remains a valid, readable subtitle file.

---

## Features

- Standalone CLI: translate one `.srt` to another
- Local inference via `llama.cpp` `llama-server` (no cloud)
- Sentence-aware grouping to avoid subtitle boundary corruption
- Windowed translation with context groups around focus groups
- Concurrent windows for improved throughput (`--concurrency`, pair with `llama-server --parallel`)
- Robust JSON parsing + repair (notably for inner quotes inside `"line"`)
- Duration-aware split-back when a translation spans multiple subtitle positions
- Cue-level reflow: wrapping respects the original cue line count
- Progress reporting: ETA, and optional tokens/sec (if the server returns timing stats)

---

## How it works (high level)

1. Parse `.srt` into cues (timestamps + lines)
2. Flatten cues into internal items (positions) while remembering how to rebuild cues later
3. Group items into sentence-safe translation units
4. Create translation windows (focus groups + surrounding context groups)
5. Translate each window via a text-completion endpoint, requiring structured JSON output
6. Parse/repair JSON, validate coverage, split translations back into original positions
7. Rebuild cues and wrap text to the original cue’s line count, then write output `.srt`

---

## Reliability: automatic fallback to smaller windows

If a translation request fails (JSON parse error, validation error, timeout, etc.), YASTS automatically retries with a **smaller focus window**.  
This is especially useful with smaller-context models or occasional model “format drift”.

Behavior (high level):
- Try the full window first.
- On failure, retry up to `max_retries_per_window`.
- If failures continue and the window contains multiple groups, YASTS shrinks the focus chunk (typically halving it) and retries.
- In the worst case, it falls back to translating **one group at a time**.

You can control this with:
- `--max-retries-per-window`
- `--shrink-focus-on-retry`
- `--max-window-chars`

---

## Endpoint compatibility

YASTS is implemented against the `llama.cpp` **`/completion`** HTTP endpoint.

It should also work with other **OpenAI-style completion endpoints** (not chat-completions) if they:
- accept a prompt string
- return a completion text payload

This has not been thoroughly tested outside `llama.cpp`, so compatibility reports/PRs are welcome.

---

## Requirements

- Python 3.10+ (3.11+ recommended)
- A running completion endpoint (tested with `llama.cpp` `llama-server`)

Python dependencies are lightweight (see `requirements.txt`).

---

## Install

Virtualenv + pip:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Ubuntu/Debian alternative (system package):

    sudo apt update
    sudo apt install -y python3-httpx python3-charset-normalizer

---

## Performance (fast)

YASTS is designed to be **fast** as well as correct.

On a system with **2× RTX 5060 Ti 16GB**, the following `llama.cpp` server parameters were tested to work well with **TranslateGemma 27B** (Q6_K) and YASTS running with `--concurrency 4`:

    llama-server \
      -m /models/translategemma-27b-it.Q6_K.gguf \
        --host 0.0.0.0 \
        --port 8080 \
        --ctx-size 10240 \
        --n-gpu-layers 999 \
        --cache-ram 0 \
        --flash-attn on \
        --split-mode layer \
        --tensor-split 1,1 \
        --cont-batching \
        --batch-size 4096 \
        --ubatch-size 1024 \
        --parallel 4 \
        --threads 4

---

## Usage

Translate an SRT file:

    python3 yasts.py \
      input.en.srt \
      output.fi.srt \
      --src-lang English \
      --tgt-lang Finnish \
      --url http://127.0.0.1:8080/completion \
      --concurrency 4

Notes:
- `--parallel 4` (server) + `--concurrency 4` (client) enables 4 in-flight translation windows at a time.
- If your model has a smaller effective context window than your server `-c`, reduce the client `--max-window-chars` to avoid truncation.

---

### Batch mode (folder crawler)

`yasts_folder.py` scans a directory tree for source-language `.srt` files and translates them by invoking `yasts.py` per file.  
It skips items that already have a target subtitle, can optionally require a corresponding video file, and supports `--dry-run`.

Any arguments after `--` are passed through verbatim to `yasts.py`:

    python3 yasts_folder.py /path/to/media --src-tag en --tgt-tag fi --skip-hi --require-video -- \
      --src-lang English --tgt-lang Finnish --url http://127.0.0.1:8080/completion --concurrency 4

---

## Defaults and tuning

YASTS ships with sane defaults in `config.py`, and all of them can be overridden via CLI parameters.

Key defaults:

Logging
- `log_level=INFO`

Server / prompt
- `llama_completion_url=http://127.0.0.1:8671/completion`
- `prompt_template=gemma3` (templates: `gemma3 | llama3 | qwen3`)
- `http_timeout_s=120.0`

Sampling
- `n_predict=2048`, `temperature=0.1`, `top_p=0.90`, `repeat_penalty=1.0`

Grouping (sentence-ish units)
- `use_phrase_grouping=True`
- `max_group_lines=8`, `max_group_chars=360`
- `min_group_text_chars=10`, `min_group_words=2`

Split-back / display shaping
- `split_max_line_chars=42`, `min_chunk_chars=10`

Windowing / batching
- `max_window_chars=2000`
- `context_pre_groups=2`, `context_post_groups=2`
- `max_retries_per_window=2`, `shrink_focus_on_retry=True`
- `concurrency=1` (override to match server parallelism)

Default languages
- `src_lang=English`, `tgt_lang=Finnish`

Tuning tips:
- If your model has a small context window, reduce `--max-window-chars`.
- If you want better consistency, increase `--context-pre-groups` / `--context-post-groups` (but watch context limits).
- For speed, increase `--concurrency` and make sure your server has `--parallel` at least as high.

---

## Debugging

If a window repeatedly fails (even at size 1), run with:

    python -m yasts.py --log-level DEBUG ...

YASTS logs:
- which group IDs were in the focus window
- prompt size estimates
- the first part of the model output
- whether failure was JSON parse or output validation

Common issue: malformed JSON from the model  
Example:

    {"line":"... mutta "valta" on ..."}

YASTS includes a targeted repair step that escapes quotes inside `"line"` fields automatically.

---

## License

MIT © 2026 [ArtoRu](https://github.com/artoru). See `LICENSE`.

---

## Contributing

Issues and PRs welcome:
- better wrapping heuristics
- improved prompt templates for additional models
- performance profiling results and tuning tips
