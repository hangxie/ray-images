#!/usr/bin/env python3
"""Narrate a book-length UTF-8 text file into a chaptered m4b using serve_tts.

  ./txt2m4b.py book.txt --dry-run               # show the detected parts and chapters, synthesize nothing
  ./txt2m4b.py book.txt --title 朱雀记 --author 猫腻
  ./txt2m4b.py book.txt --parts                 # one m4b per part
  ./txt2m4b.py book.txt --part 2                # only part 2
  ./txt2m4b.py book.txt --split book/           # write the detected chapters as files to check and fix by hand
  ./txt2m4b.py book/ --title 朱雀记              # narrate a directory of chapter files

Each non-empty line is a paragraph. A short line starting like a part heading (第一部, Part 1) or a chapter heading
(第一章, Chapter 1, 序章, 后记, Prologue) starts one; see --part-regex/--chapter-regex.

A directory input holds PPP-CCCC.txt files with the chapter title on the first line. An optional PPP-0000.txt names
the part and is read before its first chapter; 000-0000.txt is front matter.

Paragraphs are spread over the configured endpoints (see tts_client.py). Each chapter is encoded to HE-AAC under the
work directory and reused while its text and settings are unchanged, so an interrupted run resumes.
"""

import argparse
import concurrent.futures
import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile
import wave

import tts_client

NUM = r"[0-9０-９一二三四五六七八九十百千万零〇两]+"
END = r"(?=$|[\s:：.、])"
PART_REGEX = rf"(第{NUM}[部卷集篇]|(Part|Book|Volume)\s+\w+){END}"
CHAPTER_REGEX = (
    rf"(第{NUM}[章回节]|Chapter\s+\w+|序章?|楔子|引子|前言|内容简介|尾声|后记|番外|Prologue|Epilogue|Afterword){END}"
)
# Longer lines are prose even when they start like a heading, e.g. "第一部分的比赛结束了，……".
MAX_HEADING_CHARS = 40


def heading(pattern, text):
    return len(text) <= MAX_HEADING_CHARS and pattern.match(text)


def split_book(path, part_re, chapter_re):
    """Return chapters as dicts with part, title and paragraphs. Headings are spoken as part of the chapter."""
    chapters, pending, part, found = [], [], None, False
    for text in tts_client.paragraphs(path):
        if heading(part_re, text):
            # Text under a part heading with no chapter heading becomes a chapter named after the part.
            if part is not None and len(pending) > 1:
                chapters.append({"part": part, "title": part, "paragraphs": pending})
                pending = []
            part = text.rstrip(":：")
            pending.append(text)
        elif heading(chapter_re, text):
            # Front matter and a part heading are read at the start of the chapter that follows them.
            found = True
            chapters.append(
                {"part": part, "title": text.rstrip(":："), "paragraphs": [*pending, text], "lead": len(pending)}
            )
            pending = []
        elif chapters and not pending:
            chapters[-1]["paragraphs"].append(text)
        else:
            pending.append(text)
    if pending:
        chapters.append({"part": part, "title": part or pending[0], "paragraphs": pending})
    if not found:
        raise SystemExit(
            "error: no chapter headings found; put each heading on its own line, "
            'e.g. "第一章 开篇" or "Chapter 1", or pass --chapter-regex'
        )
    # Front matter before the first part heading belongs to the first part.
    first = next((c["part"] for c in chapters if c["part"]), None)
    for c in chapters:
        if c["part"]:
            break
        c["part"] = first
    return chapters


def group(chapters):
    """Group consecutive chapters of the same part into (part name, chapters) pairs."""
    parts: list[tuple[str | None, list[dict]]] = []
    for c in chapters:
        if not parts or parts[-1][0] != c["part"]:
            parts.append((c["part"], []))
        parts[-1][1].append(c)
    return parts


CHAPTER_FILE = re.compile(r"(\d+)-(\d+)\.txt")


def read_dir(path):
    """Return (part name, chapters) pairs from a directory of PPP-CCCC.txt files; see the module docstring."""
    files: dict[tuple[int, int], str] = {}
    for name in os.listdir(path):
        if not name.endswith(".txt"):
            continue
        m = CHAPTER_FILE.fullmatch(name)
        if not m:
            raise SystemExit(f"error: {os.path.join(path, name)}: chapter files are named PPP-CCCC.txt")
        key = (int(m[1]), int(m[2]))
        if key in files:
            raise SystemExit(f"error: {path}: {files[key]} and {name} have the same part and chapter number")
        files[key] = name
    if not any(c for _, c in files):
        raise SystemExit(f"error: {path} has no PPP-CCCC.txt chapter files")
    parts: list[tuple[str | None, list[dict], int]] = []
    lead: list[str] = []
    for p in sorted({p for p, _ in files}):
        name = None
        chapters: list[dict] = []
        for c in sorted(c for q, c in files if q == p):
            file = os.path.join(path, files[p, c])
            text = list(tts_client.paragraphs(file))
            if not text:
                raise SystemExit(f"error: {file} is empty")
            if c == 0:
                # Part 000 is front matter, so its 0000 file names no part.
                name, lead = (text[0].rstrip(":：") if p else None), lead + text
            else:
                chapters.append({"title": text[0].rstrip(":："), "paragraphs": lead + text})
                lead = []
        if p and not chapters:
            raise SystemExit(f"error: {path}: part {p} has no chapter files")
        if parts and parts[-1][2] == 0:
            # Chapters before the first part heading belong to the first part.
            chapters[:0] = parts.pop()[1]
        parts.append((name, chapters, p))
    for name, chapters, _ in parts:
        for chapter in chapters:
            chapter["part"] = name
    return [(name, chapters) for name, chapters, _ in parts if chapters]


def write_dir(parts, path, part_re):
    """Write parts as the files read_dir reads; text before a chapter heading goes to a PPP-0000.txt."""
    try:
        os.makedirs(path, exist_ok=True)
        if os.listdir(path):
            raise SystemExit(f"error: {path} is not empty")
    except OSError as exc:
        raise SystemExit(f"error: {exc}") from exc
    # A part has one PPP-0000.txt, so text before a later chapter (front matter, say) starts a new part.
    split: list[list[list]] = []
    for _, chs in parts:
        split.append([[], []])
        for c in chs:
            lead, body = c["paragraphs"][: c.get("lead", 0)], c["paragraphs"][c.get("lead", 0) :]
            if lead and split[-1][1]:
                split.append([[], []])
            split[-1][0] += lead
            split[-1][1].append(body)
    # The book's front matter, up to its first part heading, goes to part 000 and names no part.
    lead = split[0][0]
    k = next((k for k, text in enumerate(lead) if heading(part_re, text)), len(lead))
    front, split[0][0] = lead[:k], lead[k:]
    if not split[0][0] and len(split) > 1:
        split[0][0] = front
    else:
        split.insert(0, [front, []])
    count = 0
    for i, (lead, chs) in enumerate(split):
        for j, text in enumerate([lead, *chs]):
            if text:
                with open(os.path.join(path, f"{i:03d}-{j:04d}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(text) + "\n")
                count += 1
    print(f"Wrote {count} files to {path}")


def tag(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "default=nw=1:nk=1", path],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def span(path):
    """Return a chapter file's length in seconds including its encoder priming, which a stream copy keeps."""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "packet=pts_time,duration_time",
            "-of",
            "csv=p=0",
            path,
        ],
        text=True,
    )
    packets = [line.split(",") for line in out.split()]
    return float(packets[-1][0]) + float(packets[-1][1]) - float(packets[0][0])


def assemble(jobs, label, args):
    """Join a chapter's paragraph WAVs, each followed by the pause, into 16-bit mono PCM; return it and its rate."""
    pcm, rate = io.BytesIO(), None
    for i, job in enumerate(jobs, 1):
        with wave.open(job.path) as w:
            if (w.getnchannels(), w.getsampwidth()) != (1, 2) or rate not in (None, w.getframerate()):
                raise SystemExit(f"error: unexpected audio format from the service in {label} paragraph {i}")
            rate = w.getframerate()
            pcm.write(w.readframes(w.getnframes()))
        os.remove(job.path)
        pcm.write(bytes(2 * round(args.pause * rate)))
    return pcm.getvalue(), rate


def encode(pcm, rate, path, comment, codec, args):
    """Encode one chapter into an m4a whose comment tag records its settings and text."""
    temp = path + ".tmp.m4a"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                str(rate),
                "-ac",
                "1",
                "-i",
                "-",
                *codec,
                "-ac",
                "1",
                "-ar",
                args.sample_rate,
                "-b:a",
                args.bitrate,
                "-metadata",
                f"comment={comment}",
                "-f",
                "ipod",
                temp,
            ],
            input=pcm,
            check=True,
        )
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


def escape(value):
    # FFMETADATA treats '=', ';', '#', '\\' and newlines as special
    return "".join("\\" + ch if ch in "=;#\\\n" else ch for ch in str(value))


def aac_encoder():
    """HE-AAC needs AudioToolbox (macOS) or libfdk_aac; FFmpeg's built-in aac encoder is LC-only."""
    encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
    if re.search(r"^\s*A\S*\s+aac_at\s", encoders, re.M):
        return ["-c:a", "aac_at", "-profile:a", "4"], "aac_at HE-AAC"
    if re.search(r"^\s*A\S*\s+libfdk_aac\s", encoders, re.M):
        return ["-c:a", "libfdk_aac", "-profile:a", "aac_he"], "libfdk_aac HE-AAC"
    raise SystemExit("error: FFmpeg has neither aac_at nor libfdk_aac for HE-AAC")


def build(chapters, files, title, output, audio_comment, args):
    meta = [
        ";FFMETADATA1",
        f"title={escape(title)}",
        f"album={escape(args.title)}",
        "genre=Audiobook",
        "comment=" + escape(audio_comment),
    ]
    if args.author:
        meta += [f"artist={escape(args.author)}", f"album_artist={escape(args.author)}"]
    meta.append("")
    multi_part = len({c["part"] for c in chapters}) > 1
    # Stream copy keeps each chapter's encoder priming, which its duration excludes, so chapter marks use packet
    # spans; file durations would drift per chapter.
    lines, start, part, elapsed = [], 0, object(), 0.0
    for chapter, path in zip(chapters, files, strict=True):
        seconds = span(path)
        elapsed += seconds
        end = round(elapsed * 1000)
        # In a multi-part file, mark where each part starts on its first chapter.
        name = chapter["title"]
        if multi_part and chapter["part"] not in (part, None, name):
            name = f"{chapter['part']} {name}"
        part = chapter["part"]
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}", f"title={escape(name)}", ""]
        start = end
        lines += ["file '" + path.replace("'", "'\\''") + "'", f"duration {seconds}"]
    with tempfile.TemporaryDirectory() as tmp:
        listing, metadata = os.path.join(tmp, "list.txt"), os.path.join(tmp, "meta.txt")
        with open(listing, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        with open(metadata, "w", encoding="utf-8") as f:
            f.write("\n".join(meta))
        temp = os.path.join(os.path.dirname(output), "." + os.path.basename(output) + ".tmp")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    listing,
                    "-i",
                    metadata,
                    "-map",
                    "0:a",
                    "-map_metadata",
                    "1",
                    "-map_chapters",
                    "1",
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    "-f",
                    "ipod",
                    temp,
                ],
                check=True,
            )
            os.replace(temp, output)
        finally:
            if os.path.exists(temp):
                os.remove(temp)
    print(
        f"Created: {output}: {len(chapters)} chapters, {start / 3600000:.1f} h, "
        f"{os.path.getsize(output) / 2**20:.0f} MB"
    )


def main():
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="UTF-8 text file, one paragraph per line, or a directory of PPP-CCCC.txt files")
    ap.add_argument("output", nargs="?", help="m4b path (default: INPUT with .m4b); with --parts, the name prefix")
    ap.add_argument("--title", help="book title (default: the input file name)")
    ap.add_argument("--author", help="author, written to the artist tags")
    ap.add_argument("--parts", action="store_true", help="one m4b per part")
    ap.add_argument("--part", type=int, help="only this part number (1-based)")
    ap.add_argument("--dry-run", action="store_true", help="print the detected parts and chapters and exit")
    ap.add_argument(
        "--split", metavar="DIR", help="write the detected chapters into DIR as PPP-CCCC.txt files and exit"
    )
    ap.add_argument("--part-regex", default=PART_REGEX, help="pattern matched at the start of a part heading")
    ap.add_argument("--chapter-regex", default=CHAPTER_REGEX, help="pattern matched at the start of a chapter heading")
    ap.add_argument("--workdir", help="chapter audio cache, kept for reruns (default: OUTPUT with .work)")
    tts_client.add_arguments(ap)
    ap.add_argument("--voice", default=env("VOICE", "hang-en"))
    ap.add_argument("--language", default=env("LANGUAGE", "auto"))
    ap.add_argument("--speed", type=float, default=float(env("SPEED", "1.0")))
    ap.add_argument(
        "--pause",
        type=float,
        default=float(env("PAUSE_SECONDS", "0.5")),
        help="silence after each paragraph in seconds",
    )
    ap.add_argument("--bitrate", default=env("M4B_BITRATE", "16k"), help="AAC bitrate (default: 16k)")
    ap.add_argument(
        "--sample-rate",
        default=env("M4B_SAMPLE_RATE", "24000"),
        help="output sample rate in Hz (default: 24000, the model's native rate)",
    )
    args = ap.parse_args()

    stem = os.path.splitext(args.input.rstrip("/"))[0]
    args.output = os.path.abspath(args.output or stem + ".m4b")
    args.title = args.title or os.path.basename(stem)
    workdir = os.path.abspath(args.workdir or os.path.splitext(args.output)[0] + ".work")
    if args.pause < 0:
        ap.error("--pause must be non-negative")
    endpoints = tts_client.load_endpoints(args, ap.error)

    if os.path.isdir(args.input):
        if args.split:
            ap.error("--split needs a text file input")
        parts = read_dir(args.input)
    else:
        parts = group(split_book(args.input, re.compile(args.part_regex, re.I), re.compile(args.chapter_regex, re.I)))
    if args.split:
        write_dir(parts, args.split, re.compile(args.part_regex, re.I))
        return
    chapters = [c for _, chs in parts for c in chs]
    if args.part is not None and not 1 <= args.part <= len(parts):
        ap.error(f"--part must be between 1 and {len(parts)}")

    if args.dry_run:
        for i, (name, chs) in enumerate(parts, 1):
            print(f"Part {i}: {name or '(none)'}")
            for c in chs:
                chars = sum(len(p) for p in c["paragraphs"])
                print(f"  {c['title']}  ({len(c['paragraphs'])} paragraphs, {chars} chars)")
        total = sum(len(p) for c in chapters for p in c["paragraphs"])
        print(f"{len(parts)} parts, {len(chapters)} chapters, {total} chars")
        return

    request = {"voice": args.voice, "language": args.language, "speed": args.speed, **tts_client.SETTINGS}
    pool = tts_client.Pool(endpoints, request)
    replies = pool.probe()
    tts_client.check_speed(replies, args.speed, ap.error)
    settings = {
        "model": tts_client.models(replies),
        "voice": args.voice,
        "language": args.language,
        "speed": args.speed,
        "pause": args.pause,
        **tts_client.SETTINGS,
    }
    codec, encoder = aac_encoder()
    audio_comment = ";".join(f"{k}={v}" for k, v in settings.items())
    audio_comment += f";encoder={encoder} mono {args.sample_rate} Hz {args.bitrate}"

    selected = [(i, name, chs) for i, (name, chs) in enumerate(parts, 1) if args.part in (None, i)]
    os.makedirs(workdir, exist_ok=True)
    files = {}
    names = {id(c): f"{i:03d}-{j:03d}.m4a" for i, _, chs in selected for j, c in enumerate(chs, 1)}
    todo = [c for _, _, chs in selected for c in chs]
    # Reuse cached chapters with the same settings and text; the model isn't compared, endpoints name it differently.
    pending = []
    for k, c in enumerate(todo, 1):
        path = os.path.join(workdir, names[id(c)])
        files[id(c)] = path
        digest = hashlib.sha256("\n".join(c["paragraphs"]).encode()).hexdigest()
        comment = f"{audio_comment};text={digest}"
        if not (os.path.exists(path) and tts_client.without_model(tag(path)) == tts_client.without_model(comment)):
            pending.append((f"[{k}/{len(todo)}] {c['title']}", c, path, comment))
    if pending:
        print(f"{len(todo) - len(pending)} of {len(todo)} chapters cached; synthesizing {len(pending)}")
        # one queue across chapters; each is encoded when done, holding at most two chapters of PCM
        with tempfile.TemporaryDirectory() as tmp, concurrent.futures.ThreadPoolExecutor(1) as encoder_pool:
            chapter_jobs = []
            for k, (label, c, _, _) in enumerate(pending):
                count = len(c["paragraphs"])
                chapter_jobs.append(
                    [
                        tts_client.Job(text, os.path.join(tmp, f"{k}-{i}.wav"), f"{label} [{i}/{count}]")
                        for i, text in enumerate(c["paragraphs"], 1)
                    ]
                )
            pool.start([job for jobs in chapter_jobs for job in jobs])
            encoding = None
            try:
                for (label, _, path, comment), jobs in zip(pending, chapter_jobs, strict=True):
                    pool.wait(jobs)
                    pcm, rate = assemble(jobs, label, args)
                    if encoding:
                        encoding.result()
                    encoding = encoder_pool.submit(encode, pcm, rate, path, comment, codec, args)
                if encoding:
                    encoding.result()
            except BaseException:
                pool.stop("stopped")
                raise
        pool.summary()

    if args.parts or args.part:
        base = os.path.splitext(args.output)[0]
        for i, name, chs in selected:
            label = (name or f"part {i}").replace("：", "_").replace(":", "_").replace("/", "_").replace(" ", "_")
            build(
                chs,
                [files[id(c)] for c in chs],
                f"{args.title}：{name}" if name else args.title,
                f"{base}_{i:02d}_{label}.m4b",
                audio_comment,
                args,
            )
    else:
        build(chapters, [files[id(c)] for c in chapters], args.title, args.output, audio_comment, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
