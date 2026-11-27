#!/usr/bin/env python3
"""Narrate a UTF-8 text file into one MP3 using serve_tts.

  ./txt2mp3.py notes.txt                        # writes notes.mp3
  ./txt2mp3.py notes.txt narration.mp3 --voice hang-zh --language chinese

Each non-empty line is a paragraph, synthesized separately, kept at the model's native mono 24 kHz, and followed by
--pause seconds of silence. Paragraphs are spread over every endpoint in the endpoint config (see tts_client.py). The
MP3's comment tag records the server model, voice, language, speed, pause and sampling settings.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import wave

import tts_client


def main():
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="UTF-8 text file, one paragraph per line")
    ap.add_argument("output", nargs="?", help="MP3 path (default: INPUT with .mp3); replaced on success")
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
    ap.add_argument("--bitrate", default=env("BITRATE", "128k"), help="MP3 bitrate (default: 128k)")
    ap.add_argument(
        "--workdir",
        default=env("WORKDIR"),
        help="parent directory for temporary paragraph audio (default: TMPDIR or /tmp)",
    )
    args = ap.parse_args()

    output = args.output or os.path.splitext(args.input)[0] + ".mp3"
    if not os.path.isfile(args.input) or not os.access(args.input, os.R_OK):
        ap.error(f"input file is not readable: {args.input}")
    if os.path.exists(output) and os.path.samefile(args.input, output):
        ap.error("output must differ from input")
    if not os.path.isdir(os.path.dirname(os.path.abspath(output))) or os.path.isdir(output):
        ap.error(f"output must be a file in an existing directory: {output}")
    if args.pause < 0:
        ap.error("--pause must be non-negative")
    if not re.fullmatch(r"[1-9][0-9]*k", args.bitrate):
        ap.error("--bitrate must be a number of kbps such as 128k")
    endpoints = tts_client.load_endpoints(args, ap.error)

    texts = list(tts_client.paragraphs(args.input))
    if not texts:
        ap.error("no paragraphs found")
    print(f"Found {len(texts)} paragraphs")

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
    comment = ";".join(f"{k}={v}" for k, v in settings.items())

    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
    # Paragraph audio is never reused: input text and voice may change between runs.
    with tempfile.TemporaryDirectory(prefix="txt2mp3.", dir=args.workdir) as tmp:
        jobs = [
            tts_client.Job(text, os.path.join(tmp, f"{i}.wav"), f"[{i}/{len(texts)}]")
            for i, text in enumerate(texts, 1)
        ]
        pool.start(jobs)
        try:
            pool.wait(jobs)
        except BaseException:
            pool.stop("stopped")
            raise
        pool.summary()

        print("Combining audio...")
        temp = os.path.join(os.path.dirname(os.path.abspath(output)), "." + os.path.basename(output) + ".tmp")
        try:
            with subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "s16le",
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    "-i",
                    "-",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    args.bitrate,
                    "-metadata",
                    f"comment={comment}",
                    "-f",
                    "mp3",
                    temp,
                ],
                stdin=subprocess.PIPE,
            ) as ffmpeg:
                assert ffmpeg.stdin is not None  # stdin=PIPE
                for i, job in enumerate(jobs, 1):
                    with wave.open(job.path) as w:
                        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 24000):
                            ffmpeg.kill()
                            raise SystemExit(f"error: unexpected audio format from the service in paragraph {i}")
                        ffmpeg.stdin.write(w.readframes(w.getnframes()))
                    ffmpeg.stdin.write(bytes(2 * round(args.pause * 24000)))
                ffmpeg.stdin.close()
            if ffmpeg.returncode:
                raise SystemExit(f"error: ffmpeg failed with status {ffmpeg.returncode}")
            os.replace(temp, output)
        finally:
            if os.path.exists(temp):
                os.remove(temp)

    duration = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", output], text=True
    )
    print(f"Created: {output}")
    print(f"Settings: {comment}")
    print(f"Duration: {float(duration) / 60:.1f} minutes")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
