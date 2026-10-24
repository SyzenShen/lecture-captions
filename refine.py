#!/usr/bin/env python3
"""Re-transcribe a saved lecture recording after class, for accurate notes.

Live captions have to be quick, so they listen in short pieces. After class there is no hurry: this decodes the
whole recording at once with a stronger model, which on a real 7-minute lecture got within the noise floor of
Whisper large-v3 (13.4 % distance, versus 19-26 % for the small live model).

    ./refine.py transcripts/2026-10-23_1051.wav --prompt-file prompts/my_course.txt
"""
import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live  # noqa: E402


def session_start(path):
    """Clock time the recording started, from a name like 2026-10-23_1051.wav (else now)."""
    m = re.search(r"(\d{4})-(\d\d)-(\d\d)_(\d\d)(\d\d)", os.path.basename(path))
    return datetime(*map(int, m.groups())) if m else datetime.now()


def load_audio(path):
    """16 kHz mono samples. A recording whose session was killed has an unfinished WAV header: read it raw then."""
    from faster_whisper.audio import decode_audio
    try:
        audio = decode_audio(path, sampling_rate=live.SR)
    except Exception:
        audio = np.zeros(0, dtype=np.float32)
    if path.lower().endswith(".wav") and len(audio) < 0.9 * (os.path.getsize(path) - 44) // 2:
        audio = np.fromfile(path, dtype=np.int16, offset=44).astype(np.float32) / 32768.0
    return audio


def entries_from_segments(segments, started):
    """Turn (start, end, text) segments into notes entries, one per sentence, with clock times.

    A segment usually holds several sentences; they share its time span in proportion to their length.
    """
    entries = []
    for start, end, text in segments:
        text = text.strip()
        if not text or live.NON_ENGLISH.search(text) or live.DEGENERATE.search(text) or live.HALLUCINATION.fullmatch(text):
            continue
        sentences = live.split_sentences(text) or [text]
        total, done = max(1, sum(len(s.split()) for s in sentences)), 0
        for s in sentences:
            at = started + timedelta(seconds=start + (end - start) * done / total)
            done += len(s.split())
            entries.append({"ts": at.strftime("%H:%M:%S"), "en": s, "zh": "", "q": live.extract_question(s)})
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="audio saved by a live session, e.g. transcripts/2026-10-23_1051.wav")
    ap.add_argument("--model", default="large-v3-turbo", help="large-v3-turbo (default, ~1.6 GB), small.en (fast), large-v3 ...")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompt-file", help="topic vocabulary, as for the live captions")
    ap.add_argument("--translator", choices=["argos", "claude", "none"], default="argos")
    ap.add_argument("--md-dir", help="where to write the notes (default: the same folder as the live notes)")
    ap.add_argument("--out", help="output file (default: <md-dir>/<recording name>.refined.md)")
    a = ap.parse_args()

    if a.prompt_file:
        with open(os.path.expanduser(a.prompt_file), encoding="utf-8") as f:
            a.prompt = " ".join(f.read().split())
    prompt = live.clean_prompt(a.prompt)
    audio = load_audio(a.recording)
    if len(audio) < live.SR:
        sys.exit(f"nothing to transcribe in {a.recording}")
    minutes = len(audio) / live.SR / 60
    print(f"{os.path.basename(a.recording)}: {minutes:.1f} min of audio, model {a.model}", flush=True)

    from faster_whisper import WhisperModel
    model = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=a.threads)
    t0 = time.time()
    segs, _ = model.transcribe(audio, language="en", beam_size=5, vad_filter=True, initial_prompt=prompt or None,
                               condition_on_previous_text=False,
                               temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4)   # retry when the output loops
    segments, total = [], len(audio) / live.SR
    for s in segs:
        segments.append((s.start, s.end, s.text))
        print(f"\r  {100 * s.end / total:3.0f}%  ({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    started = session_start(a.recording)
    entries = entries_from_segments(segments, started)
    if a.translator != "none":
        translate = live.make_translator(a.translator)
        for i, e in enumerate(entries, 1):
            try:
                e["zh"] = live.clean_zh(translate(e["en"]), e["en"])
            except Exception as ex:                 # the notes are still useful without a translation
                e["zh"] = f"(translation error: {ex})"
            if i % 25 == 0:
                print(f"\r  translating {i}/{len(entries)}", end="", flush=True)
        print()

    md_dir = os.path.expanduser(a.md_dir) if a.md_dir else live.MD_DIR
    out = a.out or os.path.join(md_dir, os.path.splitext(os.path.basename(a.recording))[0] + ".refined.md")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(live.render_markdown(entries, started, prompt))
    print(f"{len(entries)} sentences -> {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
