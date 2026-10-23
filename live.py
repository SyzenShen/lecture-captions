#!/usr/bin/env python3
"""Live English lecture captions (Whisper) with Chinese translation.

  ./run.sh                      mic -> English + 中文 (offline)
  ./run.sh --list-devices       show audio inputs
  ./run.sh --device 3           pick an input device
  ./run.sh --model base.en      faster / less accurate
  ./run.sh --translator claude  better Chinese (needs ANTHROPIC_API_KEY, sends text to API)
  ./run.sh --translator none    English only
  ./run.sh --file lecture.mp3   transcribe a recording instead of the mic
"""
import argparse
import collections
import os
import queue
import re
import shutil
import sys
import threading
import time
from datetime import datetime

import numpy as np

from questions import extract_question

SR = 16000
BLOCK = 512                       # 32 ms
HERE = os.path.dirname(os.path.abspath(__file__))
MD_DIR = os.environ.get("CAPTIONS_NOTES_DIR") or os.path.join(HERE, "notes")   # where the Markdown notes (press s) are saved
ARGOS_DIR = os.path.expanduser(os.environ.get("ARGOS_ZH_DIR") or "~/.local/share/argos-translate/packages/translate-en_zh-1_9")


# ---------------------------------------------------------------- translators
LOOP = re.compile(r"(.)\1{3,}")           # one character repeated 4+ times: a translation/recognition loop


def clean_zh(zh, en):
    """Last line of defence for any translator: squash character runs and cap runaway length."""
    zh = LOOP.sub(lambda m: m.group(1) * 2, zh)
    zh = re.sub(r"\s*[(（]英语[)）]", "", zh)       # the offline model tags pure-number sentences with "(英语)"
    limit = int(len(en) * 1.8) + 12
    return zh if len(zh) <= limit else zh[:limit].rstrip("，。、 ") + "…"

class NoTranslator:
    def __call__(self, text):
        return ""


class ArgosTranslator:
    """Offline en->zh using the Argos model through ctranslate2 (no torch needed)."""

    def __init__(self):
        import ctranslate2
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=ARGOS_DIR + "/sentencepiece.model")
        self.tr = ctranslate2.Translator(ARGOS_DIR + "/model", device="cpu",
                                         compute_type="int8", intra_threads=2)

    def _one(self, sent, **kw):
        toks = self.sp.encode(sent, out_type=str)
        # a sentence never needs more than ~2x its own length in tokens; without this bound a loop runs to 256 tokens
        out = self.tr.translate_batch([toks], beam_size=4, max_decoding_length=len(toks) * 2 + 10, **kw)[0].hypotheses[0]
        return self.sp.decode(out).replace("\u2581", " ").strip()

    def __call__(self, text):
        sents = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]
        if not sents:
            return ""
        parts = []
        for sent in sents:
            zh = self._one(sent)
            if LOOP.search(zh):                 # same character 4+ times in a row: the model is looping, so retry once
                zh = self._one(sent, repetition_penalty=1.2)
            parts.append(zh)
        zh = "".join(parts)
        for a, b in ((".", "。"), (",", "，"), ("?", "？"), ("!", "！"), (":", "：")):
            zh = zh.replace(a, b)
        return zh


class ClaudeTranslator:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.history = collections.deque(maxlen=3)

    def __call__(self, text):
        ctx = "\n".join(self.history)
        msg = self.client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=500,
            system=("You translate live lecture transcripts from English to Simplified Chinese. "
                    "Output only the translation of the NEW text. Keep it natural and concise; "
                    "for technical terms give the Chinese term followed by the English in parentheses "
                    "on first use. The transcript is auto-generated and may contain recognition errors; "
                    "use context to fix obvious ones." + (f" Lecture topic: {PROMPT}" if PROMPT else "")),
            messages=[{"role": "user",
                       "content": f"Previous context:\n{ctx or '(none)'}\n\nNEW text:\n{text}"}],
        )
        self.history.append(text)
        return msg.content[0].text.strip()


def make_translator(name):
    if name == "none":
        return NoTranslator()
    if name == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set. export it first, or use --translator argos.")
        return ClaudeTranslator()
    if not os.path.exists(os.path.join(ARGOS_DIR, "model", "model.bin")):
        print("⚠ Offline translation model not found (run ./setup.sh to install it) - showing English only.\n"
              "⚠ 未找到离线翻译模型（运行 ./setup.sh 安装），本次只显示英文。", flush=True)
        return NoTranslator()
    return ArgosTranslator()


# ---------------------------------------------------------------- output
class Output:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.path = path
        self.md_path = os.path.join(MD_DIR, os.path.basename(path).replace(".txt", ".md"))
        self.topic = ""
        self.started = datetime.now()
        self.entries = []          # dicts: id (= index), ts, en, zh ("" until translated), q (question sentence or "")
        self.printed_id = None     # the sentence whose English was printed last (its Chinese belongs right under it)
        self.lock = threading.Lock()
        self.subs = []             # web clients (queues) receiving live events
        self.partial_shown = False
        self.paused = False
        self.toggle_pause = None   # set by main() in live mode
        self.show_zh = True        # toggled with the 'c' key; the Chinese is still translated and saved when hidden

    def broadcast(self, ev):
        for sub in list(self.subs):
            sub.put(ev)

    def _clear_partial(self):
        if self.partial_shown:
            self.broadcast({"type": "partial", "text": ""})
            sys.stdout.write("\r\033[2K")
            self.partial_shown = False

    def partial(self, en):
        """Unfinished sentence: dim gray, single line, replaced by the final version."""
        self.broadcast({"type": "partial", "text": en})
        width = shutil.get_terminal_size().columns
        sys.stdout.write("\r\033[2K\033[90m" + ("… " + en)[-(width - 2):] + "\033[0m")
        sys.stdout.flush()
        self.partial_shown = True

    def emit(self, en, zh=""):
        """Show an English sentence right away; its translation is attached later with set_zh()."""
        self._clear_partial()
        ts = datetime.now().strftime("%H:%M:%S")
        q = extract_question(en)
        with self.lock:
            sid = len(self.entries)
            self.entries.append({"id": sid, "ts": ts, "en": en, "zh": zh, "q": q})
            self.broadcast({"type": "final", "id": sid, "ts": ts, "en": en, "zh": zh, "q": q})
        print(f"\n\033[90m{ts}\033[0m  {en}")
        if zh and self.show_zh:
            print(f"          \033[1m{zh}\033[0m", flush=True)   # bold, default (black) colour
        self.printed_id = sid
        self.f.write(f"[{ts}] {en}\n")
        if zh:
            self.f.write(f"         {zh}\n")
        self.f.flush()
        return sid

    def set_zh(self, sid, zh):
        """The translation of sentence `sid` is ready."""
        with self.lock:
            self.entries[sid]["zh"] = zh
            en = self.entries[sid]["en"]
            self.broadcast({"type": "zh", "id": sid, "zh": zh})
        if self.show_zh:
            self._clear_partial()
            lead = "          " if self.printed_id == sid else f"          \033[90m↳ {en[:28]}…\033[0m "
            print(f"{lead}\033[1m{zh}\033[0m", flush=True)
        self.f.write(f"         {zh}\n")
        self.f.flush()

    def save_md(self):
        """Write everything captured so far to a Markdown file (overwrites the previous snapshot)."""
        with self.lock:
            entries = list(self.entries)
        lines = [f"# 课堂记录 {self.started:%Y-%m-%d %H:%M}", ""]
        if self.topic:
            lines += [f"> 主题提示：{self.topic}", ""]
        lines += [f"共 {len(entries)} 句 · 保存于 {datetime.now():%H:%M:%S}", "", "---", ""]
        for e in entries:
            lines.append(f"**{e['ts']}** {e['en']}  ")
            if e["zh"]:
                lines.append(e["zh"])
            lines.append("")
        asked = [(e["ts"], e["q"]) for e in entries if e["q"]]
        if asked:                                        # a review list of what the lecturer asked
            lines += ["---", "", f"## 课堂提问（{len(asked)}）", ""] + [f"- **{ts}** {q}" for ts, q in asked] + [""]
        os.makedirs(os.path.dirname(self.md_path), exist_ok=True)
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return len(entries)

    def toggle_zh(self):
        self.show_zh = not self.show_zh
        self._clear_partial()
        msg = "中文已显示" if self.show_zh else "中文已隐藏（仍在后台翻译并保存，按 c 重新显示）"
        print(f"\n\033[33m● {msg}\033[0m", flush=True)

    def save_md_notice(self):
        n = self.save_md()
        self._clear_partial()
        print(f"\n\033[32m✓ 已保存 {n} 句 → {self.md_path}\033[0m", flush=True)


# ---------------------------------------------------------------- display of tentative text
class Display:
    """The grey "still being worked on" line: the unfinished sentence (from the accurate recogniser) followed by
    quick drafts of speech that the accurate recogniser has not reached yet. Drafts are keyed by the start time of
    the segment they belong to, so the accurate result can replace exactly its own draft."""

    def __init__(self, out):
        self.out = out
        self.pending = ""
        self.drafts = {}
        self.lock = threading.Lock()

    def _show(self):
        shown = " ".join([self.pending] + [d for _, d in sorted(self.drafts.items())]).strip()
        if shown:
            self.out.partial(shown)
        else:
            self.out._clear_partial()

    def set_pending(self, text):
        with self.lock:
            self.pending = text
            self._show()

    def set_draft(self, key, text):
        with self.lock:
            self.drafts[key] = text
            self._show()

    def refined(self, upto):
        """The accurate recogniser has finished the segment starting at `upto`: it and any earlier draft are obsolete."""
        with self.lock:
            for k in [k for k in self.drafts if k <= upto]:
                del self.drafts[k]
            self._show()                  # repaint now: a segment with no usable result must not leave its draft behind

    def clear(self):
        with self.lock:
            self.pending, self.drafts = "", {}
            self._show()


def draft_loop(model, seg, display, every, stop):
    """Quick tentative text for the segment being spoken right now, refreshed every `every` seconds."""
    last_key, last_len = None, 0
    while not stop.wait(every):
        snap = seg.snapshot()
        if snap is None:
            continue
        audio, start = snap
        if len(audio) < SR:                       # less than a second of speech: nothing worth showing yet
            continue
        key = round(start, 2)
        if key == last_key and len(audio) - last_len < SR // 2:
            continue                              # barely any new audio since the last draft
        text = transcribe(model, audio, "", beam=1)
        last_key, last_len = key, len(audio)
        if len(text) >= 2 and not HALLUCINATION.fullmatch(text):
            display.set_draft(key, text)


# ---------------------------------------------------------------- worker
SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")
HALLUCINATION = re.compile(r"[\W_]*(you|thank you\.?|thanks for watching\.?)?[\W_]*", re.I)
GAP_COMPLETE = 0.8     # merge a following chunk only if silence in between is shorter than this ...
GAP_UNFINISHED = 3.0   # ... or this, when the text so far is obviously unfinished ("... how", "... and the")
MAX_PENDING = 25.0     # seconds of speech before we force a sentence out
DANGLING = {"a", "an", "the", "and", "or", "but", "so", "of", "to", "in", "on", "at", "for", "with", "by",
            "from", "that", "which", "who", "how", "what", "why", "when", "where", "if", "because", "as",
            "is", "are", "was", "were", "will", "can", "would", "should", "about", "into", "than", "then"}


PROMPT = ""     # topic / vocabulary hint, set by --prompt
BEAM = 5


def shorten_encoder_context(min_frames=1500, step=500):
    """faster-whisper zero-pads every chunk to 30 s before the encoder, so a 5 s chunk wastes ~80% of the
    compute. Pad to >=15 s instead. Measured on ~1000 words of noisy test speech: same word error rate
    (13.0% vs 13.2%), ~35% less CPU. (Padding to only 10 s was measurably worse, so don't go lower.)"""
    import math
    import faster_whisper.transcribe as ft
    orig = ft.pad_or_trim
    ft.pad_or_trim = lambda arr, length=3000, *, axis=-1: orig(
        arr, min(3000, max(min_frames, math.ceil(arr.shape[axis] / step) * step)), axis=axis)


def normalise(audio):
    """Bring quiet (distant-mic) speech up to a sensible level; Whisper degrades on very low input."""
    rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-9
    gain = min(0.06 / rms, 12.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32) if gain > 1.2 else audio


DEGENERATE = re.compile(r"(.)\1{7,}|\S{30,}")     # 'ooooohhhhhh...' style loops, or one giant 'word'
NON_ENGLISH = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af\uff00-\uffef]|\{\\|\\fn")


def _words(t):
    return re.sub(r"[^a-z0-9 ]", " ", t.lower().replace("-", " ")).split()


def is_prompt_echo(text, prompt, seg=None):
    """On silence/noise Whisper tends to repeat its prompt back; drop output that is just a piece of it.
    A 1-2 word output made only of prompt words ("Ewald.", "RMSD.") is dropped too, but only when Whisper was unsure
    about it, so a real one-word answer that happens to be in the prompt survives."""
    if not prompt:
        return False
    w = _words(text)
    pw = _words(prompt)
    if len(w) >= 3 and " ".join(w) in " ".join(pw):
        return True
    if seg is not None and 1 <= len(w) <= 2 and all(x in set(pw) for x in w):
        return seg.avg_logprob < -0.5 or seg.no_speech_prob > 0.15
    return False


def clean_prompt(p):
    """small.en only understands English: strip anything else and say so, instead of letting it leak into captions."""
    raw = p.strip()
    ascii_p = re.sub(r"\s+", " ", re.sub(r"[^\x00-\x7f]+", " ", raw)).strip()
    if raw and len(re.findall(r"[A-Za-z]{2,}", ascii_p)) < 2:
        print("⚠ --prompt 里没有可用的英文内容，已忽略。请写英文课程名和术语，例如："
              '--prompt "Biology lecture: protein folding, amino acid, hemoglobin"', flush=True)
        return ""
    if ascii_p != raw:
        print(f"⚠ --prompt 里的非英文字符已去掉，实际使用：{ascii_p!r}", flush=True)
    return ascii_p


def transcribe(model, audio, context, beam=None):
    prompt = PROMPT.strip() or None
    segs, _ = model.transcribe(
        normalise(audio), language="en", beam_size=beam or BEAM, vad_filter=True,
        condition_on_previous_text=False, initial_prompt=prompt, no_speech_threshold=0.6,
        max_new_tokens=int(len(audio) / SR * 7) + 20,   # a repetition loop would otherwise run to 448 tokens (~10 s)
        temperature=0.0,   # no retry-at-higher-temperature fallback: it can make one chunk take 5x longer
    )
    keep = []
    for sg in segs:
        # low-confidence output on noisy audio is almost always a hallucination ("So.", "Thank you.")
        if sg.avg_logprob < -1.0 or (sg.no_speech_prob > 0.5 and sg.avg_logprob < -0.6):
            if os.environ.get("LIVE_DEBUG"):
                print(f"[dbg] dropped {sg.text!r} logprob={sg.avg_logprob:.2f} nospeech={sg.no_speech_prob:.2f}",
                      file=sys.stderr)
            continue
        seg_text = sg.text.strip()
        # this is an English-only model: CJK characters or subtitle markup like {\fn...} are always hallucinations
        if NON_ENGLISH.search(seg_text) or DEGENERATE.search(seg_text) or sg.compression_ratio > 2.4 or is_prompt_echo(seg_text, prompt, sg):
            if os.environ.get("LIVE_DEBUG"):
                print(f"[dbg] dropped non-English/prompt-echo {seg_text!r}", file=sys.stderr)
            continue
        keep.append(seg_text)
    text = " ".join(keep).strip()
    # collapse "the structure of the structure of the structure of" style loops
    return re.sub(r"(\b[\w' ,]{4,60}?\b)(?:[ ,]+\1\b){2,}", r"\1", text, flags=re.I)


def looks_unfinished(text):
    t = text.rstrip()
    if t.endswith(("...", ",", ";", ":", "-")):
        return True
    last = re.sub(r"\W+", "", t.split()[-1]).lower()
    return last in DANGLING


def join(prev, new):
    """Glue a continuation onto the previous chunk (only called when the pause between them was short,
    so it is a continuation; Whisper capitalises/punctuates every chunk on its own)."""
    if not prev:
        return new
    prev = re.sub(r"[.\u2026,;:\-]+$", "", prev.rstrip())
    if new[:1].isupper() and not re.match(r"I\b|[A-Z]{2}", new):
        new = new[0].lower() + new[1:]
    return prev + " " + new


def split_sentences(text):
    return [x for x in re.split(r"(?<=[.?!])\s+", text.strip()) if x]


def worker(q, model, translate, out, seg, display):
    context, last = "", ""
    text, secs, end_t = "", 0.0, 0.0     # unfinished tail waiting for a continuation, and when its speech ended

    # translation runs in its own thread so it never delays the next recognition
    tq = queue.Queue()

    def translator_loop():
        while True:
            item = tq.get()
            if item is None:
                return
            sid, t = item
            try:
                zh = clean_zh(translate(t), t)
            except Exception as e:  # keep captions flowing if translation fails
                zh = f"(translation error: {e})"
            out.set_zh(sid, zh)

    tt = threading.Thread(target=translator_loop, daemon=True)
    tt.start()

    def emit(t):
        nonlocal last
        t = t.strip()
        if t.endswith("..."):
            t = t[:-3].rstrip() + "."
        elif t and t[-1] not in ".?!":
            t += "."
        if not t or t == last or HALLUCINATION.fullmatch(t):
            return
        last = t
        tq.put((out.emit(t), t))      # the English shows at once; the translator thread adds the Chinese when ready

    def finish():
        nonlocal text, secs
        emit(text)
        text, secs = "", 0.0
        display.set_pending("")

    def gap_limit():
        return GAP_UNFINISHED if looks_unfinished(text) else GAP_COMPLETE

    def run():
        nonlocal context, text, secs, end_t
        while True:
            try:
                item = q.get(timeout=0.2 if text else None)
            except queue.Empty:
                # nothing new yet: give up waiting once the silence (or the next utterance's start) is too far off
                started = seg.speech_start if seg.speaking else None
                if (started if started is not None else seg.t) - end_t > gap_limit():
                    finish()
                continue
            if item is None:
                finish()
                return
            if item == "flush":
                finish()
                display.clear()
                continue
            audio, start_t, e_t, forced = item
            if os.environ.get('LIVE_DEBUG'):
                print(f'[dbg] chunk {start_t:.1f}-{e_t:.1f}s gap={start_t - end_t:.2f}', file=sys.stderr)
            if text and start_t - end_t > gap_limit():
                finish()
            t0 = time.time()
            new = transcribe(model, audio, context)
            if os.environ.get('LIVE_DEBUG'):
                print(f'[dbg]   waited {seg.t - e_t - 0:.1f}s after speech end, transcribe {time.time() - t0:.1f}s '
                      f'for {len(audio) / SR:.1f}s audio, forced={forced}', file=sys.stderr)
            display.refined(round(start_t, 2))     # the accurate result replaces this segment's quick draft
            if len(new) < 2 or HALLUCINATION.fullmatch(new):
                continue
            context += " " + new          # raw Whisper output keeps its casing/punctuation (good prompt style)
            text, secs, end_t = join(text, new), secs + len(audio) / SR, e_t
            # every finished sentence goes out right away; only the last, possibly unfinished one waits
            parts = split_sentences(text)
            if len(parts) > 1:
                for sent in parts[:-1]:
                    emit(sent)
                text = parts[-1]
                secs = 0.0
            if secs > MAX_PENDING:
                finish()
                continue
            # a substantial sentence that ended on a real pause is done: don't wait for a possible continuation
            if not forced and not looks_unfinished(text) and len(text.split()) >= 8:
                finish()
                continue
            display.set_pending(text)

    try:
        run()
    finally:
        tq.put(None)
        tt.join(timeout=30)


# ---------------------------------------------------------------- segmentation
ABS_MIN = 0.0012      # -58 dBFS: lowest level ever treated as speech
START_RATIO = 3.0     # speech starts when a block is ~9.5 dB above the room-noise floor
KEEP_RATIO = 2.2      # ...and continues while it stays ~7 dB above it (tuned on a real quiet lecture recording)


class Segmenter:
    """Adaptive-energy utterance cutter: ends a chunk on a pause or at max length."""

    def __init__(self, pause=0.6, max_len=5.0, min_len=0.5):
        self.pause_blocks = int(pause * SR / BLOCK)
        self.max_blocks = int(max_len * SR / BLOCK)
        self.min_blocks = int(min_len * SR / BLOCK)
        self.energies = collections.deque(maxlen=int(10 * SR / BLOCK))
        self.buf, self.buf_e, self.silent, self.speaking = [], [], 0, False
        self.pre = collections.deque(maxlen=8)   # ~250 ms lead-in, as (block, energy)
        self.t = 0.0                             # stream clock, seconds of audio seen so far
        self.speech_start = 0.0

    def feed(self, block):
        self.t += BLOCK / SR
        e = float(np.sqrt(np.mean(block ** 2)))
        self.energies.append(e)
        self.n = getattr(self, "n", 0) + 1
        if self.n % 16 == 1:                  # a percentile every 16 blocks (0.5 s) is plenty, and saves wake-up work
            self.floor = min(float(np.percentile(self.energies, 10)), 0.008)   # cap: speech-at-startup must not raise it
        floor = self.floor
        # Thresholds are RELATIVE to the measured room noise, so a quiet recording (a teacher whose speech sits at
        # -45 dBFS) is judged the same as a loud one. The old fixed 0.006 floor (-44 dBFS) put such speech right on the
        # threshold: only ~10% of it counted as speech, giving a cut every 2.5 s and lost quiet words.
        start_thr = max(ABS_MIN, floor * START_RATIO)
        keep_thr = max(ABS_MIN * 0.7, floor * KEEP_RATIO)      # hysteresis: once talking, tolerate dips
        is_speech = e > (keep_thr if self.speaking else start_thr)
        if not self.speaking:
            self.pre.append((block, e))
            if is_speech:
                self.speaking, self.silent = True, 0
                self.speech_start = self.t - len(self.pre) * BLOCK / SR
                self.buf = [b for b, _ in self.pre]
                self.buf_e = [x for _, x in self.pre]
                self.pre.clear()
            return None
        self.buf.append(block)
        self.buf_e.append(e)
        self.silent = 0 if is_speech else self.silent + 1
        if self.silent >= self.pause_blocks:
            return self.flush()
        if len(self.buf) >= self.max_blocks:
            return self.split_quietest()
        return None

    def snapshot(self):
        """(audio so far, start time) of the segment being spoken right now, or None. Safe to call from another thread."""
        if not self.speaking or not self.buf:
            return None
        return np.concatenate(list(self.buf)), self.speech_start

    def reset(self):
        """Forget any half-heard utterance (used when resuming after a pause)."""
        self.buf, self.buf_e, self.silent, self.speaking = [], [], 0, False
        self.pre.clear()

    def split_quietest(self):
        """Too long without a pause: cut at the quietest ~30 ms in the last 1.3 s, not mid-word."""
        lo = len(self.buf) - int(1.3 * SR / BLOCK)
        i = lo + int(np.argmin(self.buf_e[lo:]))
        dt = BLOCK / SR
        chunk, start = np.concatenate(self.buf[:i + 1]), self.speech_start
        end = start + (i + 1) * dt
        self.buf, self.buf_e = self.buf[i + 1:], self.buf_e[i + 1:]
        self.speech_start, self.silent = end, 0
        return (chunk, start, end, True)

    def flush(self):
        """Return (audio, start_s, end_s, forced) for the finished utterance, or None if too short."""
        chunk = np.concatenate(self.buf) if self.buf else None
        long_enough = len(self.buf) - self.silent >= self.min_blocks
        end = self.t - self.silent * BLOCK / SR
        self.buf, self.buf_e, self.silent, self.speaking = [], [], 0, False
        return (chunk, self.speech_start, end, False) if (chunk is not None and long_enough) else None


VAD_START, VAD_KEEP = 0.5, 0.35   # Silero speech-probability thresholds: start talking / keep talking (hysteresis)
VAD_WINDOW_BLOCKS = 94            # ~3 s of context behind each VAD decision
VAD_STEP_BLOCKS = 4               # decide every 4 blocks (~0.13 s): finer than the 0.26 s I tested, a bit more CPU
# A fast lecturer rarely pauses 0.6 s, so segments always ran to the length limit and the first sentence of each one
# appeared a whole segment late. The longer a segment gets, the shorter the pause that ends it: (seconds into the
# segment, pause in seconds that is now enough). Cuts land on phrase boundaries instead of mid-word.
PAUSE_RELAX = ((3.0, 0.30), (5.0, 0.15))


class SileroSegmenter(Segmenter):
    """Same interface as Segmenter, but "is someone talking?" comes from the Silero VAD model that faster-whisper
    already ships. It judges how speech-like the sound is, not how loud it is, so it works the same for a quiet
    recording (speech at -45 dBFS) and a noisy room (speech only ~10 dB above the noise), where a loudness
    threshold either missed half the speech or never saw a pause."""

    def __init__(self, pause=0.6, max_len=5.0, min_len=0.5):
        super().__init__(pause, max_len, min_len)
        from faster_whisper.vad import get_vad_model
        self.vad = get_vad_model()
        self.ring = collections.deque(maxlen=VAD_WINDOW_BLOCKS)
        self.pending = []         # (block, energy) not judged yet
        self.out = []             # finished chunks waiting to be handed over

    def feed(self, block):
        self.t += BLOCK / SR
        self.ring.append(block)
        self.pending.append((block, float(np.sqrt(np.mean(block ** 2)))))
        if len(self.pending) >= VAD_STEP_BLOCKS:
            self._judge()
        return self.out.pop(0) if self.out else None

    def _judge(self):
        n = len(self.pending)
        if not n:
            return
        probs = self.vad(np.concatenate(list(self.ring))).reshape(-1)[-n:]
        real_t = self.t
        for k, ((blk, e), p) in enumerate(zip(self.pending, probs)):
            self.t = real_t - (n - 1 - k) * BLOCK / SR          # the stream clock as it was at this block
            c = self._step(blk, e, float(p))
            if c is not None:
                self.out.append(c)
        self.t = real_t
        self.pending = []

    def _step(self, block, e, p):
        is_speech = p > (VAD_KEEP if self.speaking else VAD_START)
        if not self.speaking:
            self.pre.append((block, e))
            if is_speech:
                self.speaking, self.silent = True, 0
                self.speech_start = self.t - len(self.pre) * BLOCK / SR
                self.buf = [b for b, _ in self.pre]
                self.buf_e = [x for _, x in self.pre]
                self.pre.clear()
            return None
        self.buf.append(block)
        self.buf_e.append(e)
        self.silent = 0 if is_speech else self.silent + 1
        if self.silent >= self._needed_silence():
            return Segmenter.flush(self)
        if len(self.buf) >= self.max_blocks:
            return self.split_quietest()
        return None

    def _needed_silence(self):
        """Blocks of silence that end the segment: the full pause when it is short, less as it grows."""
        need, secs = self.pause_blocks, len(self.buf) * BLOCK / SR
        for after, pause in PAUSE_RELAX:
            if secs >= after:
                need = min(need, max(2, int(pause * SR / BLOCK)))
        return need

    def flush(self):
        """Judge what is still pending, then hand over finished chunks one per call (call until None)."""
        self._judge()
        c = Segmenter.flush(self)
        res = self.out + ([c] if c is not None else [])
        self.out = res[1:]
        return res[0] if res else None

    def reset(self):
        super().reset()
        self.ring.clear()
        self.pending, self.out = [], []



# ---------------------------------------------------------------- web UI
PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>课堂字幕</title><style>
:root{--bg:#fafaf8;--fg:#1a1a1a;--dim:#8a8a85;--bar:#ecece8;--btn:#fff;--bd:#cfcfc8;--acc:#1a56db;--ok:#177245}
@media(prefers-color-scheme:dark){:root{--bg:#18181a;--fg:#f2f2ee;--dim:#8d8d96;--bar:#242428;--btn:#2e2e33;--bd:#45454c;--acc:#7aa7ff;--ok:#5fd08f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,"PingFang SC",system-ui,sans-serif}
#top{position:sticky;top:0;z-index:2}header{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 16px;background:var(--bar);border-bottom:1px solid var(--bd)}
button{font:inherit;font-size:15px;padding:7px 14px;border-radius:8px;border:1px solid var(--bd);background:var(--btn);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--acc)}button.on{background:var(--acc);border-color:var(--acc);color:#fff}
#msg{margin-left:auto;color:var(--ok);font-size:14px}.hint{color:var(--dim);font-size:13px}
main{padding:12px 16px 90px;max-width:1100px;margin:0 auto;font-size:var(--fs,26px);line-height:1.45}
.e{margin:0 0 .9em}.ts{color:var(--dim);font-size:.5em;margin-right:.6em;font-variant-numeric:tabular-nums}
.zh{font-weight:700;margin-top:.15em}.zh.wait{color:var(--dim);font-weight:400}body.nozh .zh{visibility:hidden}   /* keep the space: English must not move */
#partial{color:var(--dim)}
.e.q{border-left:4px solid var(--acc);padding-left:.55em}
/* focus mode: only the last three sentences, bigger; the Chinese opens per sentence on click */
body.focus .e:not(.vis){display:none}body.focus main{font-size:calc(var(--fs)*1.45)}body.focus .e{cursor:pointer}
body.focus .zh{display:none}body.focus .e.open .zh,body.focus.zhall .zh{display:block;visibility:visible}
#qbar{padding:8px 16px;background:var(--bar);border-bottom:1px solid var(--bd);font-size:15px;line-height:1.35}
.qt{color:var(--dim);font-size:12px;letter-spacing:.04em}#qlist div{margin-top:3px;cursor:pointer}#qlist div:first-child{font-weight:600}
#qlist .t{color:var(--dim);font-size:12px;margin-right:.5em;font-variant-numeric:tabular-nums}.e.flash{background:rgba(26,86,219,.14)}
body.paused main{opacity:.45}#pz.on{background:#d97706;border-color:#d97706;color:#fff}
</style></head><body>
<div id="top"><header><button id="pz">⏸ 暂停</button><button id="tz">隐藏中文</button><button id="fz">专注</button><button id="sv">保存 .md</button><button id="m">A−</button><button id="p">A+</button>
<span class="hint">快捷键：p 暂停 · c 隐藏/显示中文 · f 专注 · s 保存</span><span id="msg"></span></header>
<div id="qbar" hidden><span class="qt">最近的问题</span><div id="qlist"></div></div></div>
<main id="log"><div id="partial"></div></main>
<script>
const log=document.getElementById("log"),part=document.getElementById("partial"),tz=document.getElementById("tz"),msg=document.getElementById("msg");
const fzb=document.getElementById("fz"),FOCUS_N=3;
let fs=26;try{fs=+localStorage.fs||26;if(localStorage.nozh==="1")document.body.classList.add("nozh");if(localStorage.focus==="1")document.body.classList.add("focus")}catch(e){}
function paint(){document.documentElement.style.setProperty("--fs",fs+"px");tz.textContent=document.body.classList.contains("nozh")?"显示中文":"隐藏中文";tz.classList.toggle("on",document.body.classList.contains("nozh"));
 const f=document.body.classList.contains("focus"),z=document.body.classList.contains("zhall");fzb.textContent=f?"退出专注":"专注";fzb.classList.toggle("on",f);
 if(f){tz.textContent=z?"收起中文":"显示全部中文";tz.classList.toggle("on",z)}}
function layout(){const all=[...log.querySelectorAll(".e")];all.forEach((e,i)=>e.classList.toggle("vis",i>=all.length-FOCUS_N))}
function setFocus(on){document.body.classList.toggle("focus",on);if(!on)document.body.classList.remove("zhall");try{localStorage.focus=on?"1":"0"}catch(e){}layout();paint();scrollTo(0,document.body.scrollHeight)}
function toggle(){if(document.body.classList.contains("focus")){document.body.classList.toggle("zhall");paint();return}document.body.classList.toggle("nozh");try{localStorage.nozh=document.body.classList.contains("nozh")?"1":"0"}catch(e){}paint()}
function size(d){fs=Math.min(60,Math.max(16,fs+d));try{localStorage.fs=fs}catch(e){}paint()}
function flash(t){msg.textContent=t;setTimeout(()=>msg.textContent="",5000)}
async function pause(){try{await fetch("/pause",{method:"POST"})}catch(e){flash("操作失败")}}
function setPaused(p){document.body.classList.toggle("paused",p);const b=document.getElementById("pz");b.textContent=p?"▶ 继续":"⏸ 暂停";b.classList.toggle("on",p);
 document.title=p?"⏸ 课堂字幕（已暂停）":"课堂字幕";if(p)part.textContent=""}
async function save(){try{const r=await (await fetch("/save",{method:"POST"})).json();flash("✓ 已保存 "+r.n+" 句 → "+r.path)}catch(e){flash("保存失败")}}
tz.onclick=toggle;fzb.onclick=()=>setFocus(!document.body.classList.contains("focus"));log.addEventListener("click",ev=>{if(!document.body.classList.contains("focus"))return;const e=ev.target.closest(".e");if(e)e.classList.toggle("open")});document.getElementById("pz").onclick=pause;document.getElementById("sv").onclick=save;document.getElementById("m").onclick=()=>size(-2);document.getElementById("p").onclick=()=>size(2);
addEventListener("keydown",e=>{if(e.metaKey||e.ctrlKey||e.altKey)return;if(e.key==="c")toggle();if(e.key==="f")setFocus(!document.body.classList.contains("focus"));if(e.key==="p")pause();if(e.key==="s")save()});
const qs=[];
function renderQ(){const bar=document.getElementById("qbar"),box=document.getElementById("qlist");bar.hidden=!qs.length;box.textContent="";
 qs.slice(-3).reverse().forEach(x=>{const r=document.createElement("div"),t=document.createElement("span");t.className="t";t.textContent=x.ts;r.append(t,document.createTextNode(x.q));
  r.onclick=()=>{if(document.body.classList.contains("focus"))setFocus(false);x.el.scrollIntoView({block:"center",behavior:"smooth"});x.el.classList.add("flash");setTimeout(()=>x.el.classList.remove("flash"),1500)};box.append(r)})}
function add(ev){const near=innerHeight+scrollY>document.body.scrollHeight-160;const d=document.createElement("div");d.className="e"+(ev.q?" q":"");
 const t=document.createElement("span");t.className="ts";t.textContent=ev.ts;d.append(t,document.createTextNode(ev.en));
 d.dataset.id=ev.id;const z=document.createElement("div");z.className="zh"+(ev.zh?"":" wait");z.textContent=ev.zh||"…";d.append(z);
 log.insertBefore(d,part);if(ev.q){qs.push({ts:ev.ts,q:ev.q,el:d});renderQ()}layout();if(near||document.body.classList.contains("focus"))scrollTo(0,document.body.scrollHeight)}
const es=new EventSource("/events");
es.onmessage=m=>{const ev=JSON.parse(m.data);if(ev.type==="state")return setPaused(ev.paused);if(ev.type==="zh"){const z=document.querySelector('.e[data-id="'+ev.id+'"] .zh');if(z){z.textContent=ev.zh;z.classList.remove("wait")}return}if(ev.type==="final"){part.textContent="";add(ev)}else{part.textContent=ev.text?"… "+ev.text:""}};
paint();
</script></body></html>"""


def start_web(out, port):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if self.path != "/events":
                return self._send(404, "not found", "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            sub = queue.Queue()
            with out.lock:                       # snapshot + subscribe atomically: no gaps, no duplicates
                backlog = list(out.entries)
                out.subs.append(sub)
            try:
                self.wfile.write(f"data: {json.dumps({'type': 'state', 'paused': out.paused})}\n\n".encode())
                for e in backlog:
                    ev = {"type": "final", **e}
                    self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        ev = sub.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                if sub in out.subs:
                    out.subs.remove(sub)

        def do_POST(self):
            if self.path == "/pause":
                paused = out.toggle_pause() if out.toggle_pause else False
                return self._send(200, json.dumps({"paused": paused}), "application/json")
            if self.path != "/save":
                return self._send(404, "not found", "text/plain")
            n = out.save_md()
            self._send(200, json.dumps({"n": n, "path": out.md_path}, ensure_ascii=False), "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)   # localhost only
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/"


# ---------------------------------------------------------------- keyboard
def start_key_listener(out):
    """While running: 's' (or Enter) saves the Markdown file, 'c' hides/shows the Chinese, 'p' pauses/resumes. Returns a function that restores the tty."""
    if not sys.stdin.isatty():
        return lambda: None
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)          # single keypresses, no echo; Ctrl+C still works

    def loop():
        while True:
            ch = sys.stdin.read(1)
            if ch in ("s", "S", "\n", "\r"):
                out.save_md_notice()
            elif ch in ("c", "C"):
                out.toggle_zh()
            elif ch in ("p", "P") and out.toggle_pause:
                out.toggle_pause()

    threading.Thread(target=loop, daemon=True).start()
    return lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="small.en", help="tiny.en/base.en/small.en/medium.en (default small.en)")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--translator", choices=["argos", "claude", "none"], default="argos")
    ap.add_argument("--file", help="transcribe an audio file instead of the mic")
    ap.add_argument("--realtime", action="store_true", help="with --file: play at real speed (for testing)")
    ap.add_argument("--prompt", default="", help='topic/vocabulary hint, e.g. "Immunology lecture: antibody, T cell, MHC"')
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for Whisper (2 = best energy/latency balance)")
    ap.add_argument("--full-context", action="store_true", help="disable the shortened encoder context (more CPU, same accuracy)")
    ap.add_argument("--prompt-file", help="read the topic/vocabulary prompt from a text file (see prompts/)")
    ap.add_argument("--chunk", type=float, default=8.0, help="longest audio piece per recognition, seconds (8 = measured sweet spot: ~29%% less CPU than 5, same accuracy; use 5 for lower latency)")
    ap.add_argument("--md-dir", default=None, help="where the Markdown notes (press s) are saved")
    ap.add_argument("--vad", choices=["silero", "energy"], default="silero", help="how to find pauses: silero = speech-detection model (default, robust to quiet/noisy rooms); energy = old loudness threshold")
    ap.add_argument("--draft-model", default="", help="small fast model for tentative grey text while the lecturer is still speaking, e.g. base.en (default: off)")
    ap.add_argument("--draft-every", type=float, default=1.0, help="seconds between drafts")
    ap.add_argument("--draft-threads", type=int, default=2)
    ap.add_argument("--beam", type=int, default=5, help="beam size (1 = fastest, 5 = more accurate)")
    ap.add_argument("--no-record", action="store_true", help="do not save the session audio (.wav)")
    ap.add_argument("--web", action="store_true", help="open a local page with big captions and one-click buttons")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="with --web: do not open the browser automatically")
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    global PROMPT, BEAM, MD_DIR
    if a.md_dir:
        MD_DIR = os.path.expanduser(a.md_dir)
    if a.prompt_file:
        with open(os.path.expanduser(a.prompt_file), encoding="utf-8") as f:
            a.prompt = " ".join(f.read().split())
    a.prompt = clean_prompt(a.prompt)
    PROMPT, BEAM = a.prompt, a.beam
    from faster_whisper import WhisperModel
    print(f"Loading Whisper {a.model} ...", flush=True)
    if not a.full_context:
        shorten_encoder_context()
    model = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=a.threads)
    translate = make_translator(a.translator)
    out = Output(os.path.join(HERE, "transcripts", datetime.now().strftime("%Y-%m-%d_%H%M") + ".txt"))
    out.topic = a.prompt
    if a.web:
        url = start_web(out, a.port)
        print(f"网页界面：{url}", flush=True)
        if not a.no_open:
            import webbrowser
            webbrowser.open(url)
    q = queue.Queue()
    try:
        seg = SileroSegmenter(max_len=a.chunk) if a.vad == "silero" else Segmenter(max_len=a.chunk)
    except Exception as ex:                              # never leave the class without captions
        print(f"⚠ Silero 语音检测不可用（{ex}），已退回音量断句。", flush=True)
        seg = Segmenter(max_len=a.chunk)
    display = Display(out)
    stop_drafts = threading.Event()
    if a.draft_model:
        print(f"Loading draft model {a.draft_model} ...", flush=True)
        draft_model = WhisperModel(a.draft_model, device="cpu", compute_type="int8", cpu_threads=a.draft_threads)
        threading.Thread(target=draft_loop, args=(draft_model, seg, display, a.draft_every, stop_drafts), daemon=True).start()
    t = threading.Thread(target=worker, args=(q, model, translate, out, seg, display), daemon=True)
    t.start()

    if a.file:
        from faster_whisper.audio import decode_audio
        audio = decode_audio(a.file, sampling_rate=SR)
        t_start = time.perf_counter()
        for n, i in enumerate(range(0, len(audio) - BLOCK, BLOCK)):
            if a.realtime:                    # pace by an absolute schedule: sleep() oversleeps ~6 ms per call, which drifts ~19 %
                wait = t_start + n * BLOCK / SR - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            chunk = seg.feed(audio[i:i + BLOCK])
            if chunk is not None:
                q.put(chunk)
        while (tail := seg.flush()) is not None:
            q.put(tail)
        q.put(None)
        t.join()
        stop_drafts.set()
        out.save_md()
        print(f"\nSaved: {out.md_path}")
        return

    import sounddevice as sd
    import wave
    wav = None
    if not a.no_record:
        wav = wave.open(out.path.replace(".txt", ".wav"), "wb")
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(SR)

    def cb(indata, frames, time_info, status):
        x = indata[:, 0]
        if wav is not None:
            wav.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        for i in range(0, len(x) - BLOCK + 1, BLOCK):     # the buffer is 4 blocks; analyse in 32 ms steps
            chunk = seg.feed(x[i:i + BLOCK].copy())
            if chunk is not None:
                q.put(chunk)

    dev = a.device if a.device is not None else sd.default.device[0]
    print(f"Listening on: {sd.query_devices(dev)['name']}   (Ctrl+C to stop)\n"
          f"按 s 保存为 Markdown，按 c 隐藏/显示中文，按 p 暂停/继续\n保存位置 -> {out.md_path}", flush=True)

    cap = {"stream": None}
    cap_lock = threading.Lock()

    def start_capture():
        st = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK * 4, latency="high",
                            device=a.device, callback=cb)
        st.start()
        cap["stream"] = st

    def stop_capture():
        st, cap["stream"] = cap["stream"], None
        if st is not None:
            st.stop()
            st.close()

    def toggle_pause():
        """Pause = close the microphone stream entirely (no capture, no recognition, ~0% CPU)."""
        with cap_lock:
            if not out.paused:
                stop_capture()
                while (tail := seg.flush()) is not None:      # what was said just before pausing still gets translated
                    q.put(tail)
                q.put("flush")
                out.paused = True
                out._clear_partial()
                print("\n\033[33m⏸ 已暂停：麦克风已关闭，几乎不耗电。按 p 继续。\033[0m", flush=True)
            else:
                seg.reset()
                start_capture()
                out.paused = False
                print("\n\033[33m▶ 已继续\033[0m", flush=True)
            out.broadcast({"type": "state", "paused": out.paused})
            return out.paused

    out.toggle_pause = toggle_pause
    restore = start_key_listener(out)
    try:
        start_capture()
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        with cap_lock:
            stop_capture()
        while (tail := seg.flush()) is not None:
            q.put(tail)
        q.put(None)
        t.join(timeout=30)
        stop_drafts.set()
        restore()
        if wav is not None:
            wav.close()
        out.save_md_notice()


if __name__ == "__main__":
    main()
