#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import faulthandler
import logging
import signal
from pathlib import Path

from config import Settings
from llm.client import make_llm_client
from services.reviewer import Reviewer
from subtitle.grouping import group_subtitles
from subtitle.normalize import cues_to_items, items_to_cues
from subtitle.srt_io import parse_srt, write_srt
from application.windowing import make_windows
from application.window_runner import run_window_loop, run_windows_concurrent
from application.progress import LoggingProgressReporter

logger = logging.getLogger(__name__)


async def _run(input_path: Path, output_path: Path, settings: Settings) -> None:
    cues = parse_srt(str(input_path))
    if not cues:
        raise SystemExit(f"No cues parsed from {input_path}")

    items = cues_to_items(cues)
    duration_by_pos = {it.position: it.duration_s for it in items}
    out_by_pos: dict[int, str] = {it.position: it.text for it in items}

    groups = group_subtitles(items, settings=settings)

    windows = make_windows(
        groups,
        settings.max_window_chars,
        pre_groups=0,
        post_groups=0,
    )

    client = make_llm_client(settings)
    try:
        reviewer = Reviewer(client, settings, settings.src_lang, settings.tgt_lang)
        reviewer.set_items(items)

        progress = LoggingProgressReporter(logger)

        def make_chunk_processor(focus_remaining):
            async def process_chunk(focus):
                return await reviewer.review_chunk(
                    focus,
                    out_by_pos=out_by_pos,
                    duration_by_pos=duration_by_pos,
                )
            return process_chunk

        async def process_window(idx: int, w):
            return await run_window_loop(
                w.focus,
                make_chunk_processor,
                max_retries=settings.max_retries_per_window,
                shrink_on_retry=settings.shrink_focus_on_retry,
                action="review",
            )

        reviewed = await run_windows_concurrent(
            windows,
            process_window,
            concurrency=settings.concurrency,
            total_lines=len(items),
            progress=progress,
            label="Review",
        )
    finally:
        await client.aclose()

    out_by_pos.update(reviewed)

    out_cues = items_to_cues(
        cues,
        out_by_pos,
        items,
        max_line_chars=settings.split_max_line_chars,
        reflow=True,
    )
    write_srt(str(output_path), out_cues)


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = Settings()

    ap = argparse.ArgumentParser(
        description="Review translated SRT subtitles using a second LLM pass for idiom, spelling, and wording improvements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("input", help="Input translated .srt file")
    ap.add_argument("output", help="Output reviewed .srt file")

    ap.add_argument("--src-lang", default=defaults.src_lang, help="Source language")
    ap.add_argument("--tgt-lang", default=defaults.tgt_lang, help="Target language")

    ap.add_argument("--backend", default=defaults.backend,
                    choices=["llamacpp", "vllm"],
                    help="Backend type: llamacpp (native) or vllm (OpenAI-compatible)")
    ap.add_argument("--url", default=defaults.llama_completion_url, help="LLM completion URL")
    ap.add_argument("--model", default=defaults.model_name,
                    help="Model name sent to vLLM (--served-model-name)")
    ap.add_argument("--template", default=defaults.prompt_template,
                    choices=["gemma3", "gemma4", "llama3", "qwen3"],
                    help="Prompt wrapper/chat template")

    ap.add_argument("--n-predict", type=int, default=defaults.n_predict, help="Max tokens to predict")
    ap.add_argument("--temperature", type=float, default=defaults.temperature, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=defaults.top_p, help="Nucleus sampling top-p")
    ap.add_argument("--repeat-penalty", type=float, default=defaults.repeat_penalty, help="Repeat penalty")
    ap.add_argument("--timeout", type=float, default=defaults.http_timeout_s, help="HTTP timeout seconds")

    ap.add_argument("--max-group-lines", type=int, default=defaults.max_group_lines,
                    help="Max subtitle lines per group")
    ap.add_argument("--max-group-chars", type=int, default=defaults.max_group_chars,
                    help="Max characters per group (approx)")
    ap.add_argument("--min-group-text-chars", type=int, default=defaults.min_group_text_chars,
                    help="Min chars to avoid tiny-fragment group")
    ap.add_argument("--min-group-words", type=int, default=defaults.min_group_words,
                    help="Min words to avoid tiny-fragment group")

    ap.add_argument("--split-max-line-chars", type=int, default=defaults.split_max_line_chars,
                    help="Hard wrap reviewed subtitle lines to this width")
    ap.add_argument("--min-chunk-chars", type=int, default=defaults.min_chunk_chars,
                    help="Min chars per chunk when splitting reviewed text back into lines")

    ap.add_argument("--max-window-chars", type=int, default=defaults.max_window_chars,
                    help="Max approximate characters per review request")
    ap.add_argument("--max-retries", type=int, default=defaults.max_retries_per_window,
                    help="Max retries per window before forcing shrink")
    ap.add_argument("--no-shrink", action="store_true",
                    help="Do not shrink focus span on retry")
    ap.add_argument("--concurrency", type=int, default=defaults.concurrency,
                    help="Number of concurrent review requests to run")

    ap.add_argument("--with-source", action="store_true",
                    help="Include original source text in review prompt (default: off)")

    ap.add_argument("--log-level", default=defaults.log_level,
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity")

    return ap


def settings_from_args(a: argparse.Namespace) -> Settings:
    return Settings(
        log_level=a.log_level,
        llama_completion_url=a.url,
        model_name=a.model,
        prompt_template=a.template,
        http_timeout_s=a.timeout,
        backend=a.backend,
        n_predict=a.n_predict,
        temperature=a.temperature,
        top_p=a.top_p,
        repeat_penalty=a.repeat_penalty,
        use_phrase_grouping=True,
        max_group_lines=a.max_group_lines,
        max_group_chars=a.max_group_chars,
        min_group_text_chars=a.min_group_text_chars,
        min_group_words=a.min_group_words,
        split_max_line_chars=a.split_max_line_chars,
        min_chunk_chars=a.min_chunk_chars,
        max_window_chars=a.max_window_chars,
        context_pre_groups=0,
        context_post_groups=0,
        max_retries_per_window=a.max_retries,
        shrink_focus_on_retry=not a.no_shrink,
        concurrency=a.concurrency,
        src_lang=a.src_lang,
        tgt_lang=a.tgt_lang,
        review_include_source=a.with_source,
    )


def setup_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    settings = settings_from_args(args)
    setup_logging(settings)
    faulthandler.register(signal.SIGUSR1)

    asyncio.run(_run(Path(args.input), Path(args.output), settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
