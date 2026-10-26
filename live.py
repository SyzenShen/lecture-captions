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
import os

# CTranslate2's OpenMP threads busy-wait after every call by default, which is 10-25 % of the CPU with many short calls.
# This must be set before faster_whisper is imported.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import argparse
import collections
import queue
import re
import shutil
import sys
import threading
import time
from datetime import datetime

import numpy as np

import courses
from glossary import MARK_S, Glossary, GlossaryTranslator, strip_marks, tidy_zh
from questions import extract_question

SR = 16000
BLOCK = 512                       # 32 ms
HERE = os.path.dirname(os.path.abspath(__file__))
GLOSSARY = None       # set from --glossary: fixes known recognition mistakes in the English
MD_DIR = os.environ.get("CAPTIONS_NOTES_DIR") or os.path.join(HERE, "notes")   # where the Markdown notes (press s) are saved
ARGOS_DIR = os.path.expanduser(os.environ.get("ARGOS_ZH_DIR") or "~/.local/share/argos-translate/packages/translate-en_zh-1_9")


# ---------------------------------------------------------------- translators
LOOP = re.compile(r"(.)\1{3,}")           # one character repeated 4+ times: a translation/recognition loop
PHRASE_LOOP = re.compile(r"(.{2,8}?)\1{2,}")   # a 2-8 character phrase repeated 3+ times ("相接相接相接...")
INTERJECTION = re.compile(r"(?:o+h+|a+h+|u+h+m*|h+m+|w+o+w+|e+r+m*)[\s!.?,]*", re.I)   # "Ooh!" is not worth translating


def clean_zh(zh, en):
    """Last line of defence for any translator: squash character runs and cap runaway length."""
    zh = LOOP.sub(lambda m: m.group(1) * 2, zh)
    zh = PHRASE_LOOP.sub(r"\1", zh)
    zh = re.sub(r"\s*[(（]英语[)）]", "", zh)       # the offline model tags pure-number sentences with "(英语)"
    zh = tidy_zh(zh)
    limit = int(len(en) * 1.8) + 12
    if MARK_S in zh:                  # glossary terms carry their English in brackets: the length check would cut real translations
        return zh
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
        if INTERJECTION.fullmatch(text.strip()):
            return ""
        sents = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]
        if not sents:
            return ""
        parts = []
        for sent in sents:
            zh = self._one(sent)
            if LOOP.search(zh) or PHRASE_LOOP.search(zh):       # the model is looping, so retry once
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
def spread_times(first, last, n):
    """n clock times ("HH:MM:SS") spread evenly from `first` to `last` (a block's refined sentences take the old span)."""
    def secs(t):
        h, m, sec = map(int, t.split(":"))
        return h * 3600 + m * 60 + sec
    a, b = secs(first), max(secs(first), secs(last))
    out = []
    for i in range(n):
        x = a + (b - a) * i // max(1, n - 1) if n > 1 else a
        out.append(f"{x // 3600 % 24:02d}:{x // 60 % 60:02d}:{x % 60:02d}")
    return out


def render_markdown(entries, started, topic="", saved_at=None):
    """Lecture notes as Markdown. `entries` are dicts with ts, en, zh ("" if not translated) and q (a question or "")."""
    lines = [f"# 课堂记录 {started:%Y-%m-%d %H:%M}", ""]
    if topic:
        lines += [f"> 主题提示：{topic}", ""]
    lines += [f"共 {len(entries)} 句 · 保存于 {saved_at or datetime.now():%H:%M:%S}", "", "---", ""]
    for e in entries:
        lines.append(f"**{e['ts']}** {e['en']}  ")
        if e["zh"]:
            lines.append(strip_marks(e["zh"]))
        lines.append("")
    asked = [(e["ts"], e["q"]) for e in entries if e["q"]]
    if asked:                                        # a review list of what the lecturer asked
        lines += ["---", "", f"## 课堂提问（{len(asked)}）", ""] + [f"- **{ts}** {q}" for ts, q in asked] + [""]
    return "\n".join(lines)


class Output:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.path = path
        self.md_path = os.path.join(MD_DIR, os.path.basename(path).replace(".txt", ".md"))
        self.topic = ""
        self.started = datetime.now()
        self.entries = []          # dicts in display order: id (unique), ts, en, zh ("" until translated), q (question or ""), block
        self.by_id = {}
        self.next_id = 0
        self.printed_id = None     # the sentence whose English was printed last (its Chinese belongs right under it)
        self.lock = threading.Lock()
        self.subs = []             # web clients (queues) receiving live events
        self.partial_shown = False
        self.last_drafts = []
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

    def drafts(self, sentences):
        """Draft sentences that are already complete: the web page lists them as provisional rows until the accurate text arrives."""
        if sentences != self.last_drafts:
            self.last_drafts = list(sentences)
            self.broadcast({"type": "drafts", "sentences": self.last_drafts})

    def partial(self, en, web_text=None):
        """Unfinished sentence: dim gray, single line, replaced by the final version. The web page gets only the part that is
        not a complete draft sentence (web_text), because the complete ones are listed as rows."""
        self.broadcast({"type": "partial", "text": en if web_text is None else web_text})
        width = shutil.get_terminal_size().columns
        sys.stdout.write("\r\033[2K\033[90m" + ("… " + en)[-(width - 2):] + "\033[0m")
        sys.stdout.flush()
        self.partial_shown = True

    def emit(self, en, zh="", block=None):
        """Show an English sentence right away; its translation is attached later with set_zh()."""
        self._clear_partial()
        ts = datetime.now().strftime("%H:%M:%S")
        q = extract_question(en)
        with self.lock:
            sid = self.next_id
            self.next_id += 1
            entry = {"id": sid, "ts": ts, "en": en, "zh": zh, "q": q, "block": block}
            self.entries.append(entry)
            self.by_id[sid] = entry
            self.broadcast({"type": "final", **entry})
        print(f"\n\033[90m{ts}\033[0m  {en}")
        if zh and self.show_zh:
            print(f"          \033[1m{strip_marks(zh)}\033[0m", flush=True)   # bold, default (black) colour
        self.printed_id = sid
        self.f.write(f"[{ts}] {en}\n")
        if zh:
            self.f.write(f"         {strip_marks(zh)}\n")
        self.f.flush()
        return sid

    def set_zh(self, sid, zh):
        """The translation of sentence `sid` is ready."""
        with self.lock:
            entry = self.by_id.get(sid)
            if entry is None:                 # the sentence was replaced by a refined version in the meantime
                return
            entry["zh"] = zh
            en = entry["en"]
            self.broadcast({"type": "zh", "id": sid, "zh": zh})
        zh = strip_marks(zh)              # the colour markers are for the web page only
        if self.show_zh:
            self._clear_partial()
            lead = "          " if self.printed_id == sid else f"          \033[90m↳ {en[:28]}…\033[0m "
            print(f"{lead}\033[1m{zh}\033[0m", flush=True)
        self.f.write(f"         {zh}\n")
        self.f.flush()

    def replace_block(self, block, sentences, min_ratio=0.0):
        """Swap the sentences of one block for a better version of the same speech. Returns [(id, english)] of the
        new entries (their translations are still to come), or [] when nothing was replaced (also when the new version has
        fewer than min_ratio times the old number of words: that is a decoder dropping speech, not an improvement)."""
        if not sentences:
            return []
        with self.lock:
            olds = [e for e in self.entries if e["block"] == block]
            if not olds:
                return []
            if sum(len(x.split()) for x in sentences) < min_ratio * sum(len(e["en"].split()) for e in olds):
                return []
            times = spread_times(olds[0]["ts"], olds[-1]["ts"], len(sentences))
            news = []
            for t, sent in zip(times, sentences):
                news.append({"id": self.next_id, "ts": t, "en": sent, "zh": "", "q": extract_question(sent), "block": block})
                self.next_id += 1
            at = self.entries.index(olds[0])
            self.entries = [e for e in self.entries if e["block"] != block]
            self.entries[at:at] = news
            for e in olds:
                self.by_id.pop(e["id"], None)
            self.by_id.update({e["id"]: e for e in news})
            self.broadcast({"type": "replace", "drop": [e["id"] for e in olds], "entries": news})
        self.f.write(f"[refined {olds[0]['ts']}] " + " ".join(sentences) + "\n")
        self.f.flush()
        return [(e["id"], e["en"]) for e in news]

    def save_md(self):
        """Write everything captured so far to a Markdown file (overwrites the previous snapshot)."""
        with self.lock:
            entries = list(self.entries)
        os.makedirs(os.path.dirname(self.md_path), exist_ok=True)
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(render_markdown(entries, self.started, self.topic))
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
        self.upto = -1.0             # segments starting at or before this already have their accurate text
        self.lock = threading.Lock()

    def _show(self):
        texts = [self.pending] + [d for _, d in sorted(self.drafts.items())]
        shown = " ".join(texts).strip()
        complete, tails = [], []
        for text in texts[1:]:                  # complete draft sentences become rows on the web page, the rest stays the grey line
            parts = split_sentences(text)
            if not parts:
                continue
            if SENTENCE_END.search(parts[-1]) and not looks_unfinished(parts[-1]):
                complete.extend(parts)
            else:
                complete.extend(parts[:-1])
                tails.append(parts[-1])
        self.out.drafts(complete)
        if shown:
            self.out.partial(shown, web_text=" ".join([self.pending] + tails).strip())
        else:
            self.out._clear_partial()

    def set_pending(self, text):
        with self.lock:
            self.pending = text
            self._show()

    def set_draft(self, key, text):
        with self.lock:
            if key <= self.upto:          # the accurate text arrived while this draft was being made: it would never go away
                return
            self.drafts[key] = text
            self._show()

    def has_draft(self, key):
        with self.lock:
            return key in self.drafts or key <= self.upto

    def refined(self, upto):
        """The accurate recogniser has finished the segment starting at `upto`: it and any earlier draft are obsolete."""
        with self.lock:
            self.upto = max(self.upto, upto)
            for k in [k for k in self.drafts if k <= upto]:
                del self.drafts[k]
            self._show()                  # repaint now: a segment with no usable result must not leave its draft behind

    def clear(self):
        with self.lock:
            self.pending, self.drafts = "", {}
            self._show()


def draft_loop(model, seg, display, every, stop, late=None):
    """Quick tentative text for the segment being spoken right now. The first draft comes as soon as a second of speech
    has been heard (that is what the reader waits for); after that it is refreshed at most every `every` seconds."""
    last_key, last_len, last_time = None, 0, 0.0
    while not stop.wait(0.25):
        while late is not None and not late.empty():      # a piece that ended before it ever got a draft (too short, or the loop was busy)
            audio, start = late.get_nowait()
            key = round(start, 2)
            if not display.has_draft(key):
                text = transcribe(model, audio, "", beam=1, vad=False)
                if len(text) >= 2 and not HALLUCINATION.fullmatch(text):
                    display.set_draft(key, text)
        snap = seg.snapshot()
        if snap is None:
            continue
        audio, start = snap
        if len(audio) < SR:                       # less than a second of speech: nothing worth showing yet
            continue
        key = round(start, 2)
        now = time.monotonic()
        if key == last_key and (now - last_time < every or len(audio) - last_len < SR // 2):
            continue                              # too soon, or barely any new audio since the last draft
        text = transcribe(model, audio, "", beam=1, vad=False)       # the segmenter already cut this at pauses
        last_key, last_len, last_time = key, len(audio), time.monotonic()
        if len(text) >= 2 and not HALLUCINATION.fullmatch(text):
            display.set_draft(key, text)


class BlockGrouper:
    """Groups consecutive chunks into blocks of at least `seconds` of speech. A block ends at the first real pause once it
    is long enough, or at 1.8x that length whatever happens. seconds = 0 turns it off (every chunk is block 0)."""

    def __init__(self, seconds=0.0):
        self.seconds, self.block, self.start = seconds, 0, None

    def tag(self, chunk):
        """(audio, start, end, forced) -> (audio, start, end, forced, block, ends_block)."""
        audio, start, end, forced = chunk
        if not self.seconds:
            return audio, start, end, forced, 0, False
        if self.start is None:
            self.start = start
        span = end - self.start
        block, ends = self.block, span >= self.seconds and (not forced or span >= 1.8 * self.seconds)
        if ends:
            self.block, self.start = self.block + 1, None
        return audio, start, end, forced, block, ends

    def close(self):
        """An explicit break (pause, end of the lecture): the next chunk starts a new block."""
        if self.start is not None:
            self.block, self.start = self.block + 1, None


class BlockAudio:
    """Collects the audio of each block (same grouping as BlockGrouper) and hands finished blocks to the refiner queue."""

    def __init__(self, rq):
        self.rq, self.parts, self.block = rq, [], None

    def add(self, tagged):
        audio, block, ends = tagged[0], tagged[4], tagged[5]
        self.parts.append(audio)
        self.block = block
        if ends:
            self.close()

    def close(self):
        if self.parts:
            self.rq.put((self.block, np.concatenate(self.parts)))
            self.parts = []


class Refined:
    """A better version of one block, sent to the worker so it is applied after the block's live sentences exist."""

    def __init__(self, block, sentences):
        self.block, self.sentences = block, sentences


MAX_REFINE_BACKLOG = 3     # blocks waiting for the refiner: beyond this the oldest is left as it is (the live text stays)


def refiner(rq, model, q):
    """Decode each finished block once more with a stronger model; the worker swaps the block's sentences for the result."""
    while True:
        item = rq.get()
        if item is None:
            return
        while rq.qsize() >= MAX_REFINE_BACKLOG:       # falling behind: skip the oldest rather than lag further
            skipped = rq.get()
            if skipped is None:
                return
            if os.environ.get("LIVE_DEBUG"):
                print(f"[dbg] refiner behind, skipped block {skipped[0]}", file=sys.stderr)
        block, audio = item
        t0 = time.time()
        pieces = transcribe(model, audio, "", beam=5, careful=True, as_segments=True)
        if os.environ.get("LIVE_DEBUG"):
            print(f"[dbg] refined block {block}: {len(audio) / SR:.1f}s audio in {time.time() - t0:.1f}s", file=sys.stderr)
        sentences = refined_sentences(pieces)
        if sentences:
            q.put(Refined(block, sentences))


def start_translator(translate, out):
    """Translation runs in its own thread so it never delays recognition. Put (sentence id, English) on the queue."""
    tq = queue.Queue()

    def loop():
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

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return tq, thread


# ---------------------------------------------------------------- worker
SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")
HALLUCINATION = re.compile(r"[\W_]*(you|thank you\.?|thanks for watching\.?)?[\W_]*", re.I)
GAP_COMPLETE = 0.8     # merge a following chunk only if silence in between is shorter than this ...
GAP_UNFINISHED = 3.0   # ... or this, when the text so far is obviously unfinished ("... how", "... and the")
MERGE_WAIT = 1.5       # seconds of silence after which chunks waiting to be merged are decoded anyway
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


def transcribe(model, audio, context, beam=None, careful=False, as_segments=False, vad=True):
    """careful=True is for long audio decoded once (the block refiner): retry at a higher temperature when the output
    looks like a repetition loop. The live path leaves it off because a retry can make one chunk take 5x longer."""
    prompt = PROMPT.strip() or None
    retry = dict(temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4) if careful else dict(temperature=0.0)
    segs, _ = model.transcribe(
        normalise(audio), language="en", beam_size=beam or BEAM, vad_filter=vad,
        condition_on_previous_text=False, initial_prompt=prompt, no_speech_threshold=0.6,
        max_new_tokens=int(len(audio) / SR * 7) + 20,   # a repetition loop would otherwise run to 448 tokens (~10 s)
        **retry,
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
    # collapse "the structure of the structure of the structure of" style loops
    loop = re.compile(r"(\b[\w' ,]{4,60}?\b)(?:[ ,]+\1\b){2,}", re.I)
    if as_segments:
        return [x for x in (loop.sub(r"\1", k).strip() for k in keep) if x]
    return loop.sub(r"\1", " ".join(keep).strip())


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


def refined_sentences(segment_texts, max_words=30):
    """Sentences for a refined block. Whisper sometimes returns a long stretch without any punctuation: break it at its
    own segment boundaries, and as a last resort after max_words words."""
    out, buf = [], ""
    for seg_text in segment_texts:
        buf = (buf + " " + seg_text).strip()
        parts = split_sentences(buf)
        if len(parts) > 1:
            out.extend(parts[:-1])
            buf = parts[-1]
        if buf[-1:] in ".?!" or len(buf.split()) >= max_words // 2:
            while len(buf.split()) > max_words:
                words = buf.split()
                out.append(" ".join(words[:max_words]))
                buf = " ".join(words[max_words:])
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def worker(q, model, tq, out, seg, display):
    context, last, cur_block = "", "", 0
    text, secs, end_t = "", 0.0, 0.0     # unfinished tail waiting for a continuation, and when its speech ended

    def emit(t):
        nonlocal last
        t = t.strip()
        if t.endswith("..."):
            t = t[:-3].rstrip() + "."
        elif t and t[-1] not in ".?!":
            t += "."
        if GLOSSARY:
            t = GLOSSARY.fix_english(t)
        if not t or t == last or HALLUCINATION.fullmatch(t):
            return
        last = t
        tq.put((out.emit(t, block=cur_block), t))      # the English shows at once; the translator thread adds the Chinese when ready

    def finish():
        nonlocal text, secs
        emit(text)
        text, secs = "", 0.0
        display.set_pending("")

    def gap_limit():
        return GAP_UNFINISHED if looks_unfinished(text) else GAP_COMPLETE

    waiting = []          # chunks collected for one decode: (audio, start, end, forced, block, ends, decode_now)

    def decode_waiting():
        """Decode the collected chunks as one piece: fewer, longer decodes cost about half the CPU of one decode per
        chunk (the encoder always processes >= 15 s of padded audio) and read better with more context."""
        nonlocal context, text, secs, end_t, cur_block
        if not waiting:
            return
        items, waiting[:] = list(waiting), []
        audio = np.concatenate([i[0] for i in items]) if len(items) > 1 else items[0][0]
        start_t, e_t, forced, block = items[0][1], items[-1][2], items[-1][3], items[0][4]
        ends = any(i[5] for i in items)
        if os.environ.get('LIVE_DEBUG'):
            print(f'[dbg] decode {len(items)} chunk(s) {start_t:.1f}-{e_t:.1f}s gap={start_t - end_t:.2f}', file=sys.stderr)
        if text and start_t - end_t > gap_limit():
            finish()
        cur_block = block
        t0 = time.time()
        new = transcribe(model, audio, context)
        if os.environ.get('LIVE_DEBUG'):
            print(f'[dbg]   waited {seg.t - e_t - 0:.1f}s after speech end, transcribe {time.time() - t0:.1f}s '
                  f'for {len(audio) / SR:.1f}s audio, forced={forced}', file=sys.stderr)
        display.refined(round(items[-1][1], 2))     # the accurate result replaces the quick drafts of all its chunks
        if len(new) < 2 or HALLUCINATION.fullmatch(new):
            if ends:
                finish()
            return
        context += " " + new          # raw Whisper output keeps its casing/punctuation (good prompt style)
        text, secs, end_t = join(text, new), secs + len(audio) / SR, e_t
        # every finished sentence goes out right away; only the last, possibly unfinished one waits
        parts = split_sentences(text)
        if len(parts) > 1:
            for sent in parts[:-1]:
                emit(sent)
            text = parts[-1]
            secs = 0.0
        # a block end is a sentence end (the refined version of the block replaces whole sentences); a substantial
        # sentence that ended on a real pause is done too: don't wait for a possible continuation
        done = SENTENCE_END.search(text) and not looks_unfinished(text)      # "Yes." / "That is why." need not wait for a continuation
        if secs > MAX_PENDING or ends or (not forced and not looks_unfinished(text) and (done or len(text.split()) >= 8)):
            finish()
            return
        display.set_pending(text)

    def run():
        while True:
            try:
                item = q.get(timeout=0.2 if (text or waiting) else None)
            except queue.Empty:
                # nothing new yet: give up waiting once the silence (or the next utterance's start) is too far off
                started = seg.speech_start if seg.speaking else None
                now = started if started is not None else seg.t
                if waiting and now - waiting[-1][2] > MERGE_WAIT:
                    decode_waiting()
                if text and now - end_t > gap_limit():
                    finish()
                continue
            if item is None:
                decode_waiting()
                finish()
                return
            if isinstance(item, Refined):
                if waiting and waiting[0][4] == item.block:
                    decode_waiting()
                if text and cur_block == item.block:    # the lecture ended inside this block: its last sentence is not out yet
                    finish()
                sentences = [GLOSSARY.fix_english(x) for x in item.sentences] if GLOSSARY else item.sentences
                for sid, en in out.replace_block(item.block, sentences, min_ratio=0.6):
                    tq.put((sid, en))
                continue
            if isinstance(item, str) and item == "flush":
                decode_waiting()
                finish()
                display.clear()
                continue
            waiting.append(item)
            if item[6]:
                decode_waiting()

    run()


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
PAUSE_RELAX = ((2.0, 0.30), (3.5, 0.15))     # tuned on a real lecture: first words appear 38% sooner than with ((3, .3), (5, .15))


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
# Words that show how an argument is built (contrast, cause, condition, example, sequence). The page prints them in bold so a
# long sentence is easier to follow. "and"/"or" are left out on purpose: they are everywhere and would drown the rest.
CONNECTIVES = (
    "but", "however", "although", "though", "whereas", "while", "yet", "unlike", "instead", "otherwise", "despite",
    "on the other hand", "in contrast", "by contrast", "nevertheless", "even though",
    "because", "since", "so", "so that", "such that", "therefore", "thus", "hence", "consequently", "as a result", "that is why",
    "if", "unless", "then", "in that case", "as long as", "provided that",
    "for example", "for instance", "such as", "e.g.", "in other words", "that is", "namely", "i.e.",
    "also", "moreover", "furthermore", "in addition", "additionally",
    "first", "second", "third", "finally", "next", "afterwards", "first of all", "in the end", "in summary", "to summarize",
)


# The same for the Chinese line. The first group cannot be mistaken for part of another word; the second group is only a
# connective at the start of a clause ("那么多" and "最后一个" are not).
CONNECTIVES_ZH = ("但是", "但", "然而", "不过", "虽然", "尽管", "因为", "由于", "所以", "因此", "于是", "从而", "如果", "假如", "除非", "否则",
                  "例如", "比如", "也就是说", "换句话说", "此外", "而且", "总之", "综上所述", "即使", "只要", "无论")
CONNECTIVES_ZH_CLAUSE = ("那么", "然后", "同时", "另外", "首先", "其次", "最后", "接着", "还有", "同样")


def connective_pattern_zh():
    """Regex source for Chinese connectives (Python and JavaScript). "但" alone only counts at the start of a clause."""
    free = sorted((x for x in CONNECTIVES_ZH if x != "但"), key=len, reverse=True)
    clause = sorted(CONNECTIVES_ZH_CLAUSE + ("但",), key=len, reverse=True)
    start = r"(?<![\u4e00-\u9fff])"
    return "(?:" + "|".join(free) + ")|" + start + "(?:" + "|".join(clause) + ")"


def connective_pattern():
    """Regex source (works in Python and JavaScript) matching any CONNECTIVES entry as whole words, longest first."""
    parts = sorted(CONNECTIVES, key=len, reverse=True)
    return r"(?<![\w.])(?:" + "|".join(re.escape(x).replace("\\ ", r"\s+") for x in parts) + r")(?![\w])"


PAGE = r"""<!doctype html><html lang="zh-Hans"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>课堂字幕</title><style>
:root{--term:#8b1a1a;--bg:#fafaf8;--fg:#1a1a1a;--dim:#8a8a85;--bar:#ecece8;--btn:#fff;--bd:#cfcfc8;--acc:#1a56db;--ok:#177245}
@media(prefers-color-scheme:dark){:root{--term:#ff9a9a;--bg:#18181a;--fg:#f2f2ee;--dim:#8d8d96;--bar:#242428;--btn:#2e2e33;--bd:#45454c;--acc:#7aa7ff;--ok:#5fd08f}}
*{box-sizing:border-box}
/* Chinese must name its font FIRST: with -apple-system first, Safari picks whatever CJK font the system cascade offers for a glyph, which on a
   Mac with Office installed is sometimes Microsoft YaHei. English keeps the system font. */
body{margin:0;background:var(--bg);color:var(--fg);font-family:"PingFang SC","Hiragino Sans GB","Heiti SC",sans-serif}
.e,#partial,#qlist{font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif}.zh{font-family:"PingFang SC","Hiragino Sans GB","Heiti SC",sans-serif}
#top{position:sticky;top:0;z-index:2}header{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 16px;background:var(--bar);border-bottom:1px solid var(--bd)}
button{font:inherit;font-size:15px;padding:7px 14px;border-radius:8px;border:1px solid var(--bd);background:var(--btn);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--acc)}button.on{background:var(--acc);border-color:var(--acc);color:#fff}
#msg{margin-left:auto;color:var(--ok);font-size:14px}.hint{color:var(--dim);font-size:13px}
main{padding:12px 16px 90px;max-width:1100px;margin:0 auto;font-size:var(--fs,26px);line-height:1.45}
.e{margin:0 0 .9em}.ts{color:var(--dim);font-size:.5em;margin-right:.6em;font-variant-numeric:tabular-nums}
.zh{font-weight:400;margin-top:.15em}.zh b.cn{font-weight:800}.term{color:var(--term);margin:0 .12em}.zh.wait{color:var(--dim);font-weight:400}body.nozh .zh{visibility:hidden}   /* keep the space: English must not move */
#partial,.e.prov{color:var(--fg);opacity:.62}   /* the draft: readable, but visibly not final */
.e.q{border-left:4px solid var(--acc);padding-left:.55em}
/* focus mode: only the last three sentences, bigger; the Chinese opens per sentence on click */
body.focus .e:not(.vis){display:none}body.focus main{font-size:calc(var(--fs)*1.45)}body.focus .e{cursor:pointer}
body.focus .zh{display:none}body.focus .e.open .zh,body.focus.zhall .zh{display:block;visibility:visible}
#qbar{padding:8px 16px;background:var(--bar);border-bottom:1px solid var(--bd);font-size:15px;line-height:1.35}
.qt{color:var(--dim);font-size:12px;letter-spacing:.04em}#qlist div{margin-top:3px;cursor:pointer}#qlist div:first-child{font-weight:600}
#qlist .t{color:var(--dim);font-size:12px;margin-right:.5em;font-variant-numeric:tabular-nums}.e.flash{background:rgba(26,86,219,.14)}
b.cn{font-weight:700}body.nocn b.cn{font-weight:inherit}body.paused main{opacity:.45}#pz.on{background:#d97706;border-color:#d97706;color:#fff}
</style></head><body>
<div id="top"><header><button id="pz">⏸ 暂停</button><button id="tz">隐藏中文</button><button id="fz">专注</button><button id="cn">连接词</button><button id="sv">保存 .md</button><button id="m">A−</button><button id="p">A+</button>
<span class="hint">快捷键：p 暂停 · c 隐藏/显示中文 · f 专注 · b 连接词加粗 · s 保存</span><span id="msg"></span></header>
<div id="qbar" hidden><span class="qt">最近的问题</span><div id="qlist"></div></div></div>
<main id="log"><div id="drafts"></div><div id="partial"></div></main>
<script>
const log=document.getElementById("log"),part=document.getElementById("partial"),dr=document.getElementById("drafts"),tz=document.getElementById("tz"),msg=document.getElementById("msg");
const fzb=document.getElementById("fz"),cnb=document.getElementById("cn"),FOCUS_N=3,CONN=new RegExp(__CONN__,"gi"),CONNZH=new RegExp(__CONNZH__,"g");
let fs=26;try{fs=+localStorage.fs||26;if(localStorage.nozh==="1")document.body.classList.add("nozh");if(localStorage.focus==="1")document.body.classList.add("focus");if(localStorage.nocn==="1")document.body.classList.add("nocn")}catch(e){}
function paint(){document.documentElement.style.setProperty("--fs",fs+"px");tz.textContent=document.body.classList.contains("nozh")?"显示中文":"隐藏中文";tz.classList.toggle("on",document.body.classList.contains("nozh"));cnb.classList.toggle("on",!document.body.classList.contains("nocn"));
 const f=document.body.classList.contains("focus"),z=document.body.classList.contains("zhall");fzb.textContent=f?"退出专注":"专注";fzb.classList.toggle("on",f);
 if(f){tz.textContent=z?"收起中文":"显示全部中文";tz.classList.toggle("on",z)}}
function layout(){const all=[...log.querySelectorAll(".e")];all.forEach((e,i)=>e.classList.toggle("vis",i>=all.length-FOCUS_N))}
function setFocus(on){document.body.classList.toggle("focus",on);if(!on)document.body.classList.remove("zhall");try{localStorage.focus=on?"1":"0"}catch(e){}layout();paint();scrollTo(0,document.body.scrollHeight)}
function toggle(){if(document.body.classList.contains("focus")){document.body.classList.toggle("zhall");paint();return}document.body.classList.toggle("nozh");try{localStorage.nozh=document.body.classList.contains("nozh")?"1":"0"}catch(e){}paint()}
function toggleCn(){document.body.classList.toggle("nocn");try{localStorage.nocn=document.body.classList.contains("nocn")?"1":"0"}catch(e){}paint()}
function marked(text){const f=document.createDocumentFragment();let i=0;CONN.lastIndex=0;for(let m;(m=CONN.exec(text));){if(m.index>i)f.append(text.slice(i,m.index));const b=document.createElement("b");b.className="cn";b.textContent=m[0];f.append(b);i=m.index+m[0].length}if(i<text.length)f.append(text.slice(i));return f}
function boldZh(text){const f=document.createDocumentFragment();let i=0;CONNZH.lastIndex=0;for(let m;(m=CONNZH.exec(text));){if(m.index>i)f.append(text.slice(i,m.index));const b=document.createElement("b");b.className="cn";b.textContent=m[0];f.append(b);i=m.index+m[0].length}if(i<text.length)f.append(text.slice(i));return f}
function fillZh(el,text){el.textContent="";text.split(/(\ue000[^\ue001]*\ue001)/).forEach(p=>{if(p.startsWith("\ue000")){const s=document.createElement("span");s.className="term";s.append(boldZh(p.slice(1,-1)));el.append(s)}else if(p)el.append(boldZh(p))})}
function size(d){fs=Math.min(60,Math.max(16,fs+d));try{localStorage.fs=fs}catch(e){}paint()}
function flash(t){msg.textContent=t;setTimeout(()=>msg.textContent="",5000)}
async function pause(){try{await fetch("/pause",{method:"POST"})}catch(e){flash("操作失败")}}
function setPaused(p){document.body.classList.toggle("paused",p);const b=document.getElementById("pz");b.textContent=p?"▶ 继续":"⏸ 暂停";b.classList.toggle("on",p);
 document.title=p?"⏸ 课堂字幕（已暂停）":"课堂字幕";if(p)part.textContent=""}
async function save(){try{const r=await (await fetch("/save",{method:"POST"})).json();flash("✓ 已保存 "+r.n+" 句 → "+r.path)}catch(e){flash("保存失败")}}
tz.onclick=toggle;cnb.onclick=toggleCn;fzb.onclick=()=>setFocus(!document.body.classList.contains("focus"));log.addEventListener("click",ev=>{if(!document.body.classList.contains("focus"))return;const e=ev.target.closest(".e");if(e)e.classList.toggle("open")});document.getElementById("pz").onclick=pause;document.getElementById("sv").onclick=save;document.getElementById("m").onclick=()=>size(-2);document.getElementById("p").onclick=()=>size(2);
addEventListener("keydown",e=>{if(e.metaKey||e.ctrlKey||e.altKey)return;if(e.key==="c")toggle();if(e.key==="b")toggleCn();if(e.key==="f")setFocus(!document.body.classList.contains("focus"));if(e.key==="p")pause();if(e.key==="s")save()});
const qs=[];
function renderQ(){const bar=document.getElementById("qbar"),box=document.getElementById("qlist");bar.hidden=!qs.length;box.textContent="";
 qs.slice(-3).reverse().forEach(x=>{const r=document.createElement("div"),t=document.createElement("span");t.className="t";t.textContent=x.ts;r.append(t,document.createTextNode(x.q));
  r.onclick=()=>{if(document.body.classList.contains("focus"))setFocus(false);x.el.scrollIntoView({block:"center",behavior:"smooth"});x.el.classList.add("flash");setTimeout(()=>x.el.classList.remove("flash"),1500)};box.append(r)})}
function makeEntry(ev){const d=document.createElement("div");d.className="e"+(ev.q?" q":"");
 const t=document.createElement("span");t.className="ts";t.textContent=ev.ts;d.append(t,marked(ev.en));
 d.dataset.id=ev.id;d.dataset.q=ev.q||"";const z=document.createElement("div");z.className="zh"+(ev.zh?"":" wait");z.lang="zh-Hans";if(ev.zh)fillZh(z,ev.zh);else z.textContent="…";d.append(z);return d}
function rebuildQ(){qs.length=0;document.querySelectorAll(".e.q").forEach(el=>qs.push({ts:el.querySelector(".ts").textContent,q:el.dataset.q,el}));renderQ()}
function add(ev){const near=innerHeight+scrollY>document.body.scrollHeight-160;const d=makeEntry(ev);
 log.insertBefore(d,dr);if(ev.q){qs.push({ts:ev.ts,q:ev.q,el:d});renderQ()}layout();if(near||document.body.classList.contains("focus"))scrollTo(0,document.body.scrollHeight)}
function replaceBlock(ev){const olds=ev.drop.map(id=>document.querySelector('.e[data-id="'+id+'"]')).filter(Boolean);const anchor=olds[0]||dr;
 ev.entries.forEach(n=>log.insertBefore(makeEntry(n),anchor));olds.forEach(el=>el.remove());rebuildQ();layout()}
function renderDrafts(list){const near=innerHeight+scrollY>document.body.scrollHeight-160;dr.textContent="";
 list.forEach(t=>{const d=document.createElement("div");d.className="e prov";const m=document.createElement("span");m.className="ts";m.textContent="暂定";d.append(m,t);dr.append(d)});layout();if(near||document.body.classList.contains("focus"))scrollTo(0,document.body.scrollHeight)}
const es=new EventSource("/events");
es.onmessage=m=>{const ev=JSON.parse(m.data);if(ev.type==="drafts")return renderDrafts(ev.sentences);if(ev.type==="state")return setPaused(ev.paused);if(ev.type==="replace")return replaceBlock(ev);if(ev.type==="zh"){const z=document.querySelector('.e[data-id="'+ev.id+'"] .zh');if(z){fillZh(z,ev.zh);z.classList.remove("wait")}return}if(ev.type==="final"){part.textContent="";add(ev)}else{part.textContent=ev.text?"… "+ev.text:""}};
paint();
</script></body></html>"""


def start_web(out, port):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    page = PAGE.replace("__CONN__", json.dumps(connective_pattern())).replace("__CONNZH__", json.dumps(connective_pattern_zh()))

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
                return self._send(200, page, "text/html; charset=utf-8")
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
# defaults per power situation (explicit flags always win)
PROFILES = {
    "battery": {"no_draft": True, "merge_seconds": 6.0, "threads": 1, "beam": 3, "draft_every": 1.5, "draft_threads": 1},
    "balanced": {"merge_seconds": 10.0, "threads": 1, "beam": 3, "draft_every": 1.5, "draft_threads": 1},
    "plugged": {"merge_seconds": 10.0, "threads": 2, "beam": 3, "draft_every": 1.0, "draft_threads": 2,
                "refine_model": "large-v3-turbo", "block_seconds": 15.0},
}


def startup_banner(a, course_name, power, url, md_path, mic=None):
    """What the user sees after start-up: course, profile, where the page is, the keys. Short on purpose."""
    names = {"battery": "省电：无草稿", "balanced": "平衡：带灰色草稿", "plugged": "插电：强模型逐段精校"}
    where = {"battery": "电池", "ac": "插电"}.get(power, "")
    lines = []
    if course_name:
        prompt = os.path.basename(a.prompt_file) if a.prompt_file else "无"
        gl = f" · 术语表 {len(GLOSSARY.terms)} 条/修正 {len(GLOSSARY.fixes)} 条" if GLOSSARY else ""
        lines.append(f"课程  {course_name}（提示词 {prompt}{gl}）")
    else:
        lines.append("课程  未指定，没有提示词（下次可以 ./run.sh 课程名，./run.sh --list-courses 看有哪些）")
    lines.append(f"档位  {a.profile}（{names.get(a.profile, '')}{'，' + where if where else ''}）")
    if url:
        lines.append(f"网页  {url}")
    if mic:
        lines.append(f"麦克风  {mic}")
    lines.append("按键  p 暂停 · c 隐藏/显示中文 · f 专注 · s 保存 · Ctrl+C 结束")
    lines.append(f"笔记  {md_path}")
    return "\n".join(lines)


LAST_COURSE = os.path.join(HERE, ".last_course")


def pick_course(a, all_courses):
    """Fill in a.prompt_file / a.glossary from the course named on the command line, asked for, or used last time."""
    last = courses.read_last(LAST_COURSE)
    name = a.course
    if not name and sys.stdin.isatty() and all_courses:
        print("课程（输入名字或编号，回车 = " + (f"上次的 {last}" if last else "不用提示词") + "）：")
        keys = sorted(all_courses)
        for i, key in enumerate(keys, 1):
            print(f"  {i:2d}  {courses.describe(key, all_courses[key])}")
        try:
            answer = input("> ").strip()
        except EOFError:
            answer = ""
        if answer.isdigit() and 1 <= int(answer) <= len(keys):
            name = keys[int(answer) - 1]
        else:
            name = answer or last
        if name and " " in name:                         # "ngs 3": the lesson can be typed here too
            name, a.lesson = name.split(None, 1)
    if not name:
        return None
    key, matches = courses.resolve(name, all_courses)
    if key is None:
        sys.exit(f"找不到课程“{name}”。" + (f"可能是：{', '.join(matches)}" if matches else "用 ./run.sh --list-courses 查看已有的课程。"))
    course = all_courses[key]
    if a.lesson is not None and not str(a.lesson).isdigit():
        sys.exit(f"课号应该是数字，收到“{a.lesson}”。")
    path = courses.pick_prompt(course, a.lesson)
    if path is None and a.lesson is not None:
        sys.exit(f"课程 {key} 没有第 {a.lesson} 课的提示词。已有：{', '.join(courses.lessons(course)) or '无'}")
    a.prompt_file = path
    a.glossary = list(a.glossary) + course["glossaries"]
    courses.write_last(LAST_COURSE, key)
    return key + (f" 第{a.lesson}课" if a.lesson is not None else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("course", nargs="?", help="course name (start of a file name in prompts/ or glossary/): its prompt and glossary are used; without it you are asked")
    ap.add_argument("lesson", nargs="?", help="lesson number, if the course has one prompt file per lesson (e.g. ./run.sh ngs 3)")
    ap.add_argument("--list-courses", action="store_true", help="show the courses found in prompts/ and glossary/")
    ap.add_argument("--model", default="small.en", help="tiny.en/base.en/small.en/medium.en (default small.en)")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--translator", choices=["argos", "claude", "none"], default="argos")
    ap.add_argument("--file", help="transcribe an audio file instead of the mic")
    ap.add_argument("--realtime", action="store_true", help="with --file: play at real speed (for testing)")
    ap.add_argument("--prompt", default="", help='topic/vocabulary hint, e.g. "Immunology lecture: antibody, T cell, MHC"')
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for Whisper (profile default: 1 on battery/balanced, 2 plugged; 1 thread uses less total CPU)")
    ap.add_argument("--full-context", action="store_true", help="disable the shortened encoder context (more CPU, same accuracy)")
    ap.add_argument("--prompt-file", help="read the topic/vocabulary prompt from a text file (see prompts/)")
    ap.add_argument("--chunk", type=float, default=6.0, help="longest audio piece per recognition, seconds (6: first words appear sooner than with 8, ~20%% more CPU)")
    ap.add_argument("--md-dir", default=None, help="where the Markdown notes (press s) are saved")
    ap.add_argument("--vad", choices=["silero", "energy"], default="silero", help="how to find pauses: silero = speech-detection model (default, robust to quiet/noisy rooms); energy = old loudness threshold")
    ap.add_argument("--draft-model", default="tiny.en", help="small fast model for tentative grey text while the lecturer is still speaking (default tiny.en: first words appear ~2 s after they are spoken instead of ~6 s)")
    ap.add_argument("--no-draft", action="store_true", help="no draft text: about 40%% less CPU, first words appear later")
    ap.add_argument("--draft-every", type=float, default=None, help="minimum seconds between drafts of the same segment (the first comes after 1 s of speech)")
    ap.add_argument("--draft-threads", type=int, default=None)
    ap.add_argument("--glossary", action="append", default=[], metavar="FILE",
                    help="course glossary (see glossary/example_biology.txt): right Chinese for technical terms, fixes for words Whisper keeps mishearing; can be given more than once")
    ap.add_argument("--brackets", choices=["first", "always"], default="always",
                    help="glossary terms are shown as English（中文）: every time (default), or only the first time, plain Chinese after that")
    ap.add_argument("--profile", choices=["auto"] + list(PROFILES), default=os.environ.get("CAPTIONS_PROFILE", "auto"),
                    help="battery = least CPU; balanced = + grey draft text; plugged = + a strong model re-reads every ~15 s block and replaces it (needs power); auto (default) = battery on battery, balanced when plugged in (CAPTIONS_PROFILE sets another default)")
    ap.add_argument("--refine-model", default=None, help="second pass: re-decode every block with this model (e.g. large-v3-turbo) and replace the live text; set by --profile plugged")
    ap.add_argument("--refine-threads", type=int, default=4)
    ap.add_argument("--merge-seconds", type=float, default=0.0, help="decode several consecutive pieces of speech as one, ending at a pause after this many seconds: about half the CPU and more accurate, but the accurate text comes later (0 = one decode per piece)")
    ap.add_argument("--block-seconds", type=float, default=0.0, help="length of a block for --refine-model (it ends at the first pause after this many seconds)")
    ap.add_argument("--beam", type=int, default=None, help="beam size of the accurate pass (profile default 3: no measurable loss on 8-15 s blocks, ~25 %% less CPU than 5)")
    ap.add_argument("--no-record", action="store_true", help="do not save the session audio (.wav)")
    ap.add_argument("--web", action="store_true", help="(the local page is on by default for the microphone; with --file it is only shown with this option)")
    ap.add_argument("--no-web", action="store_true", help="terminal only, no local page")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()
    a.web = not a.no_web and (a.web or not a.file)
    all_courses = courses.scan(os.path.join(HERE, "prompts"), os.path.join(HERE, "glossary"))
    shared = all_courses.pop("common", {"glossaries": []})["glossaries"]     # glossary/common_*.txt: for every course
    a.glossary = list(a.glossary) + shared
    if a.list_courses:
        for key in sorted(all_courses):
            print("  " + courses.describe(key, all_courses[key]))
        return
    course_name, power = None, courses.power_source()
    if not (a.prompt_file or a.prompt):
        course_name = pick_course(a, all_courses)
    if a.profile == "auto":
        a.profile = "battery" if power == "battery" else "balanced"
    for key, value in PROFILES[a.profile].items():          # a profile only fills in what the user did not set
        if getattr(a, key) == ap.get_default(key):
            setattr(a, key, value)
    if a.refine_model and not a.block_seconds:
        a.block_seconds = 15.0

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    global PROMPT, BEAM, MD_DIR, GLOSSARY
    if a.md_dir:
        MD_DIR = os.path.expanduser(a.md_dir)
    if a.prompt_file:
        with open(os.path.expanduser(a.prompt_file), encoding="utf-8") as f:
            a.prompt = " ".join(f.read().split())
    a.prompt = clean_prompt(a.prompt)
    PROMPT, BEAM = a.prompt, a.beam
    from faster_whisper import WhisperModel
    print("正在加载模型……", flush=True)
    if not a.full_context:
        shorten_encoder_context()
    model = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=a.threads)
    translate = make_translator(a.translator)
    if a.glossary:
        GLOSSARY = Glossary.load([os.path.expanduser(p) for p in a.glossary], always=a.brackets == "always")
        translate = GlossaryTranslator(translate, GLOSSARY)
    url = None
    out = Output(os.path.join(HERE, "transcripts", datetime.now().strftime("%Y-%m-%d_%H%M") + ".txt"))
    out.topic = a.prompt
    if a.web:
        url = start_web(out, a.port)
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
    late_drafts = None
    if a.draft_model and not a.no_draft:
        draft_model = WhisperModel(a.draft_model, device="cpu", compute_type="int8", cpu_threads=a.draft_threads)
        late_drafts = queue.Queue()
        threading.Thread(target=draft_loop, args=(draft_model, seg, display, a.draft_every, stop_drafts, late_drafts), daemon=True).start()
    tq, tt = start_translator(translate, out)
    grouper, merger = BlockGrouper(a.block_seconds), BlockGrouper(a.merge_seconds)
    rq, blocks, refine_thread = queue.Queue(), None, None
    if a.refine_model:
        try:
            refine_model = WhisperModel(a.refine_model, device="cpu", compute_type="int8", cpu_threads=a.refine_threads)
            blocks = BlockAudio(rq)
            refine_thread = threading.Thread(target=refiner, args=(rq, refine_model, q), daemon=True)
        except Exception as ex:                          # e.g. offline and not downloaded yet: captions must still work
            print(f"⚠ 精校模型无法加载（{ex}），只用实时字幕。", flush=True)
            grouper = BlockGrouper(0)

    def put_chunk(c):
        if late_drafts is not None:
            late_drafts.put((c[0], c[1]))
        tagged = grouper.tag(c)
        merged_end = merger.tag(c)[5] if a.merge_seconds else True
        q.put(tagged + (merged_end or tagged[5],))        # the worker decodes when a merge group or a refine block ends
        if blocks is not None:
            blocks.add(tagged)

    def end_block():
        grouper.close()
        merger.close()
        if blocks is not None:
            blocks.close()

    def finish_refining():
        """Let the refiner deliver what it has; nothing may be queued behind the worker's end marker."""
        end_block()
        if refine_thread is not None:
            print("正在完成最后几段的精校…", flush=True)
            rq.put(None)
            refine_thread.join(timeout=90)

    t = threading.Thread(target=worker, args=(q, model, tq, out, seg, display), daemon=True)
    t.start()
    if refine_thread is not None:
        refine_thread.start()

    if a.file:
        print(startup_banner(a, course_name, power, url, out.md_path), flush=True)
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
                put_chunk(chunk)
        while (tail := seg.flush()) is not None:
            put_chunk(tail)
        finish_refining()
        q.put(None)
        t.join()
        stop_drafts.set()
        tq.put(None)
        tt.join(timeout=30)
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
                put_chunk(chunk)

    dev = a.device if a.device is not None else sd.default.device[0]
    print(startup_banner(a, course_name, power, url, out.md_path, mic=sd.query_devices(dev)["name"]), flush=True)

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
                    put_chunk(tail)
                end_block()
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
            put_chunk(tail)
        finish_refining()
        q.put(None)
        t.join(timeout=30)
        stop_drafts.set()
        tq.put(None)
        tt.join(timeout=30)
        restore()
        if wav is not None:
            wav.close()
        out.save_md_notice()


if __name__ == "__main__":
    main()
