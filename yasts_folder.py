#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Candidate:
    srt_path: Path
    prefix: str
    is_hi: bool


def iter_candidates(root: Path, src_tag: str, hi_tag: str, skip_hi: bool) -> Iterable[Candidate]:
    """
    Source filename patterns supported (recursive):
      - prefix.<src_tag>.srt
      - prefix.<src_tag>.<hi_tag>.srt
      - prefix.<hi_tag>.<src_tag>.srt
    """
    hi_a = f".{src_tag}.{hi_tag}.srt"  # en.hi.srt
    hi_b = f".{hi_tag}.{src_tag}.srt"  # hi.en.srt
    plain = f".{src_tag}.srt"          # en.srt

    for srt_path in root.rglob("*.srt"):
        name = srt_path.name

        # more specific first
        if name.endswith(hi_a):
            if skip_hi:
                continue
            prefix = name[: -len(hi_a)]
            yield Candidate(srt_path=srt_path, prefix=prefix, is_hi=True)
        elif name.endswith(hi_b):
            if skip_hi:
                continue
            prefix = name[: -len(hi_b)]
            yield Candidate(srt_path=srt_path, prefix=prefix, is_hi=True)
        elif name.endswith(plain):
            prefix = name[: -len(plain)]
            yield Candidate(srt_path=srt_path, prefix=prefix, is_hi=False)


def parse_video_exts(values: Sequence[str]) -> tuple[str, ...]:
    """
    Accept repeated --video-ext flags or comma-separated lists.
    Examples:
      --video-ext mkv --video-ext mp4
      --video-ext "mkv,mp4,avi"
    """
    out: list[str] = []
    for v in values:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        out.extend(parts)

    seen = set()
    deduped: list[str] = []
    for e in out:
        e2 = e.lstrip(".")
        if e2 not in seen:
            seen.add(e2)
            deduped.append(e2)
    return tuple(deduped)


def has_matching_video(dir_path: Path, prefix: str, video_exts: Sequence[str]) -> bool:
    for ext in video_exts:
        ext = ext.lstrip(".")
        if (dir_path / f"{prefix}.{ext}").exists():
            return True
    return False


def find_existing_target_sub(dir_path: Path, prefix: str, tgt_tag: str, hi_tag: str) -> Path | None:
    """
    Return an existing target subtitle path (first match) if any exist:
      - prefix.<tgt>.srt
      - prefix.<hi>.<tgt>.srt
      - prefix.<tgt>.<hi>.srt
    """
    variants = (
        dir_path / f"{prefix}.{tgt_tag}.srt",
        dir_path / f"{prefix}.{hi_tag}.{tgt_tag}.srt",
        dir_path / f"{prefix}.{tgt_tag}.{hi_tag}.srt",
    )
    for p in variants:
        if p.exists():
            return p
    return None


def build_output_path(dir_path: Path, prefix: str, is_hi: bool, tgt_tag: str, hi_tag: str, ai_tag: str) -> Path:
    out_base = f"{hi_tag}.{tgt_tag}" if is_hi else f"{tgt_tag}"
    out_suffix = f"{out_base}.{ai_tag}.srt" if ai_tag else f"{out_base}.srt"
    return dir_path / f"{prefix}.{out_suffix}"


def run_translation(translator_script: Path, src_srt: Path, dst_srt: Path, translator_args: Sequence[str]) -> None:
    if not translator_script.exists():
        raise FileNotFoundError(f"Translator script not found: {translator_script}")

    cmd = [sys.executable, str(translator_script), str(src_srt), str(dst_srt), *translator_args]
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yasts_folder.py",
        add_help=False,  # we provide combined help
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Recursively scan folders for source-language SRTs and translate them using yasts.py.\n"
            "\n"
            "Source patterns:\n"
            "  - prefix.<src-tag>.srt\n"
            "  - prefix.<src-tag>.<hi-tag>.srt\n"
            "  - prefix.<hi-tag>.<src-tag>.srt\n"
            "\n"
            "Skip if any target subtitle already exists:\n"
            "  - prefix.<tgt-tag>.srt\n"
            "  - prefix.<hi-tag>.<tgt-tag>.srt\n"
            "  - prefix.<tgt-tag>.<hi-tag>.srt\n"
            "\n"
            "Passing args to yasts.py:\n"
            "  Recommended: use `--` to separate crawler args from translator args.\n"
            "    translate_folder.py <dir> [crawler args...] -- [yasts args...]\n"
            "\n"
            "  Convenience: if you forget `--`, any unknown args are treated as yasts args.\n"
        ),
    )

    p.add_argument("-h", "--help", action="store_true", help="Show this help and yasts.py help, then exit.")

    p.add_argument("directory", nargs="?", default=".", help="Root directory to scan recursively.")
    p.add_argument("--src-tag", default="en", help="Source language tag used in filenames (e.g. en, sv, de).")
    p.add_argument("--tgt-tag", default="fi", help="Target language tag used in filenames (e.g. fi, sv, de).")
    p.add_argument("--hi-tag", default="hi", help="Hearing-impaired tag used in filenames.")
    p.add_argument("--ai-tag", default="ai", help='Optional output marker tag. Use --ai-tag "" to disable.')

    p.add_argument(
        "--skip-hi",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, ignore HI source subtitles (both orderings).",
    )
    p.add_argument(
        "--require-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, only translate subtitles that have a corresponding video file (same prefix).",
    )
    p.add_argument(
        "--video-ext",
        action="append",
        default=["mkv,mp4,avi,mov,m4v,webm,ts"],
        help="Video extensions used with --require-video. Can repeat or use comma-separated list.",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of subtitles to translate (0 = no limit). Counts only translations, not skips.",
    )

    p.add_argument(
        "--translator-script",
        default=str(Path(__file__).with_name("yasts.py")),
        help="Path to yasts.py (or another translator script with compatible CLI).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not run translation; only report what would be translated or skipped.",
    )

    return p


def print_combined_help(parser: argparse.ArgumentParser, translator_script: Path) -> int:
    parser.print_help()
    print("\n--- yasts.py help ---\n")

    if not translator_script.exists():
        print(f"[!] Translator script not found: {translator_script}", file=sys.stderr)
        return 2

    proc = subprocess.run(
        [sys.executable, str(translator_script), "-h"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    return 0


def split_argv_on_double_dash(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def main() -> int:
    parser = build_parser()

    raw_argv = sys.argv[1:]
    crawler_argv, translator_argv = split_argv_on_double_dash(raw_argv)

    args, unknown = parser.parse_known_args(crawler_argv)
    translator_script = Path(args.translator_script)

    if args.help:
        return print_combined_help(parser, translator_script)

    translator_args = translator_argv + unknown

    root = Path(args.directory)
    if not root.is_dir():
        print(f"Error: Directory not found: {root}", file=sys.stderr)
        return 1

    video_exts = parse_video_exts(args.video_ext)
    limit = int(args.limit)
    if limit < 0:
        print("Error: --limit must be >= 0", file=sys.stderr)
        return 2

    print(f"Processing (recursively): {root}")
    print(
        f"Filename tags: {args.src_tag} -> {args.tgt_tag} | HI tag: {args.hi_tag} | "
        f"AI tag: {args.ai_tag if args.ai_tag else '<none>'}"
    )
    print(f"Skip HI: {args.skip_hi} | Require video: {args.require_video} | Video exts: {', '.join(video_exts)}")
    print(f"Translator: {translator_script}")
    if translator_args:
        print(f"Translator args: {' '.join(translator_args)}")
    if args.dry_run:
        print("DRY RUN: no translations will be executed")
    if limit:
        print(f"Limit: {limit} translation(s)")

    translated_count = 0

    try:
        for cand in iter_candidates(root, args.src_tag, args.hi_tag, args.skip_hi):
            if limit and translated_count >= limit:
                print(f"Limit reached ({limit}). Stopping.")
                break

            src = cand.srt_path
            dir_path = src.parent

            if args.require_video and not has_matching_video(dir_path, cand.prefix, video_exts):
                print(f"Skipping: no matching video for {src}")
                continue

            existing = find_existing_target_sub(dir_path, cand.prefix, args.tgt_tag, args.hi_tag)
            if existing is not None:
                print(f"Skipping: {existing} already exists.")
                continue

            out_path = build_output_path(dir_path, cand.prefix, cand.is_hi, args.tgt_tag, args.hi_tag, args.ai_tag)

            if out_path.exists():
                print(f"Skipping: {out_path} already exists.")
                continue

            print(f"Translating {src} -> {out_path}")
            translated_count += 1

            if args.dry_run:
                continue

            run_translation(translator_script, src, out_path, translator_args)

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"Translation command failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())