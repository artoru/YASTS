#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
import faulthandler, signal

from pathlib import Path
from typing import List

from subtitle_translator.config import Settings
from subtitle_translator.grouping import group_subtitles
from subtitle_translator.normalize import cues_to_items, items_to_cues
from subtitle_translator.pipeline import build_1to1_groups, translate_items_with_windows
from subtitle_translator.srt_io import parse_srt, write_srt


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = Settings()

    ap = argparse.ArgumentParser(
        description="Translate SRT subtitles using llama.cpp with sentence grouping + context windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("input", help="Input .srt file")
    ap.add_argument("output", help="Output .srt file")

    # Languages
    ap.add_argument("--src-lang", default=defaults.src_lang, help="Source language")
    ap.add_argument("--tgt-lang", default=defaults.tgt_lang, help="Target language")

    # Llama endpoint + prompt wrapper
    ap.add_argument("--url", default=defaults.llama_completion_url, help="llama.cpp /completion URL")
    ap.add_argument("--template", default=defaults.prompt_template, choices=["gemma3", "llama3", "qwen3"],
                    help="Prompt wrapper/chat template")

    # Sampling / generation
    ap.add_argument("--n-predict", type=int, default=defaults.n_predict, help="Max tokens to predict")
    ap.add_argument("--temperature", type=float, default=defaults.temperature, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=defaults.top_p, help="Nucleus sampling top-p")
    ap.add_argument("--repeat-penalty", type=float, default=defaults.repeat_penalty, help="Repeat penalty")
    ap.add_argument("--timeout", type=float, default=defaults.http_timeout_s, help="HTTP timeout seconds")

    # Grouping (sentence-ish units)
    ap.add_argument("--no-phrase-grouping", action="store_true",
                    help="Disable sentence-ish grouping (not recommended)")
    ap.add_argument("--max-group-lines", type=int, default=defaults.max_group_lines,
                    help="Max subtitle lines per group")
    ap.add_argument("--max-group-chars", type=int, default=defaults.max_group_chars,
                    help="Max characters per group (approx)")
    ap.add_argument("--min-group-text-chars", type=int, default=defaults.min_group_text_chars,
                    help="Min chars to avoid tiny-fragment group")
    ap.add_argument("--min-group-words", type=int, default=defaults.min_group_words,
                    help="Min words to avoid tiny-fragment group")

    # Split-back formatting
    ap.add_argument("--split-max-line-chars", type=int, default=defaults.split_max_line_chars,
                    help="Hard wrap translated subtitle lines to this width")
    ap.add_argument("--min-chunk-chars", type=int, default=defaults.min_chunk_chars,
                    help="Min chars per chunk when splitting group translations back into lines")

    # Context windowing / batching
    ap.add_argument("--max-window-chars", type=int, default=defaults.max_window_chars,
                    help="Max approximate characters for (context + focus) payload per request")
    ap.add_argument("--pre", type=int, default=defaults.context_pre_groups,
                    help="Context groups before focus")
    ap.add_argument("--post", type=int, default=defaults.context_post_groups,
                    help="Context groups after focus")
    ap.add_argument("--max-retries", type=int, default=defaults.max_retries_per_window,
                    help="Max retries per window before forcing shrink")
    ap.add_argument("--no-shrink", action="store_true",
                    help="Do not shrink focus span on retry")
    ap.add_argument("--concurrency", type=int, default=defaults.concurrency,
                help="Number of concurrent window requests to run")

    # Logging
    ap.add_argument("--log-level", default=defaults.log_level,
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity")

    return ap


def settings_from_args(a: argparse.Namespace) -> Settings:
    return Settings(
        log_level=a.log_level,
        llama_completion_url=a.url,
        prompt_template=a.template,
        http_timeout_s=a.timeout,
        n_predict=a.n_predict,
        temperature=a.temperature,
        top_p=a.top_p,
        repeat_penalty=a.repeat_penalty,
        use_phrase_grouping=not a.no_phrase_grouping,
        max_group_lines=a.max_group_lines,
        max_group_chars=a.max_group_chars,
        min_group_text_chars=a.min_group_text_chars,
        min_group_words=a.min_group_words,
        split_max_line_chars=a.split_max_line_chars,
        min_chunk_chars=a.min_chunk_chars,
        max_window_chars=a.max_window_chars,
        context_pre_groups=a.pre,
        context_post_groups=a.post,
        max_retries_per_window=a.max_retries,
        shrink_focus_on_retry=not a.no_shrink,
        concurrency=a.concurrency,
        src_lang=a.src_lang,
        tgt_lang=a.tgt_lang,
    )


def setup_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    
    # Silence very noisy dependencies when running at DEBUG
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)  # sometimes appears with HTTP/2 stacks
    logging.getLogger("asyncio").setLevel(logging.WARNING)


async def run_translate(in_path: str, out_path: str, settings: Settings) -> None:
    cues = parse_srt(in_path)
    if not cues:
        raise SystemExit(f"No cues parsed from {in_path}")
    
    faulthandler.register(signal.SIGUSR1)
    
    items, pos_map, duration_by_pos = cues_to_items(cues)

    if settings.use_phrase_grouping:
        groups = group_subtitles(items, settings=settings)
    else:
        groups = build_1to1_groups(items)

    translated_by_pos = await translate_items_with_windows(
        items=items,
        groups=groups,
        settings=settings,
        src_lang=settings.src_lang,
        tgt_lang=settings.tgt_lang,
        duration_by_pos=duration_by_pos,
    )

    out_cues = items_to_cues(
        cues,
        translated_by_pos,
        pos_map,
        max_line_chars=settings.split_max_line_chars,
        reflow=True,
    )
    
    write_srt(out_path, out_cues)


def main(argv: List[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    settings = settings_from_args(args)
    setup_logging(settings)

    asyncio.run(run_translate(str(Path(args.input)), str(Path(args.output)), settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
