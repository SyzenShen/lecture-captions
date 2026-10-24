#!/usr/bin/env python3
"""Measure cloud speech-to-text on YOUR recordings: accuracy against a reference text, and the round-trip delay per chunk.

It cuts the audio exactly like the live captioner does (Silero pauses, 6 s limit), sends each chunk to the provider and
times the answer. Nothing is sent unless you pass --yes. API keys are read from environment variables only:

    GROQ_API_KEY      whisper-large-v3-turbo on Groq
    OPENAI_API_KEY    gpt-4o-transcribe (override with OPENAI_STT_MODEL)
    DEEPGRAM_API_KEY  nova-3 (topic terms are sent as keyterms)

    GROQ_API_KEY=... python tools/cloud_eval.py clip.wav --provider groq --ref reference.txt --prompt-file prompts/x.txt --yes

WARNING: this uploads the audio to a third party. Check your university's recording rules and the provider's data policy.
"""
import argparse
import io
import json
import os
import re
import statistics
import sys
import time
import urllib.parse
import urllib.request
import uuid
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import live  # noqa: E402

PROVIDERS = {
    "groq": {"url": "https://api.groq.com/openai/v1/audio/transcriptions", "key": "GROQ_API_KEY", "kind": "openai", "model": "whisper-large-v3-turbo"},
    "openai": {"url": "https://api.openai.com/v1/audio/transcriptions", "key": "OPENAI_API_KEY", "kind": "openai",
               "model": os.environ.get("OPENAI_STT_MODEL", "gpt-4o-transcribe")},
    "deepgram": {"url": "https://api.deepgram.com/v1/listen", "key": "DEEPGRAM_API_KEY", "kind": "deepgram", "model": "nova-3"},
}


def wav_bytes(audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(live.SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def multipart(fields, filename, data):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def endpoint(p):
    """Provider URL, or the same path on CLOUD_EVAL_BASE_URL (used by the tests to point at a local fake server)."""
    base = os.environ.get("CLOUD_EVAL_BASE_URL")
    return base.rstrip("/") + urllib.parse.urlparse(p["url"]).path if base else p["url"]


def transcribe_chunk(p, key, data, prompt, terms):
    """Returns the text for one WAV chunk."""
    if p["kind"] == "openai":
        fields = {"model": p["model"], "language": "en", "response_format": "json", "temperature": "0"}
        if prompt:
            fields["prompt"] = prompt
        body, ctype = multipart(fields, "chunk.wav", data)
        req = urllib.request.Request(endpoint(p), data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": ctype})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["text"]
    query = [("model", p["model"]), ("language", "en"), ("smart_format", "true")] + [("keyterm", t) for t in terms[:100]]
    req = urllib.request.Request(endpoint(p) + "?" + urllib.parse.urlencode(query), data=data,
                                 headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"})
    res = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return res["results"]["channels"][0]["alternatives"][0]["transcript"]


def cut_like_the_captioner(audio, max_len=6.0):
    seg, chunks = live.SileroSegmenter(max_len=max_len), []
    for i in range(0, len(audio) - live.BLOCK + 1, live.BLOCK):
        c = seg.feed(audio[i:i + live.BLOCK].copy())
        if c is not None:
            chunks.append(c)
    while (c := seg.flush()) is not None:
        chunks.append(c)
    return chunks


def words(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("-", " ")).split()


def distance(ref, hyp):
    """Word-level edit distance / reference length (same measure as used for the live captioner)."""
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="a recording (wav, mp3 ...)")
    ap.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    ap.add_argument("--ref", help="reference text to measure the distance against")
    ap.add_argument("--prompt-file", help="topic vocabulary (sent as the prompt, or as Deepgram keyterms)")
    ap.add_argument("--chunk", type=float, default=6.0)
    ap.add_argument("--out", help="write the transcript here")
    ap.add_argument("--yes", action="store_true", help="really upload the audio")
    a = ap.parse_args()

    p = PROVIDERS[a.provider]
    key = os.environ.get(p["key"], "")
    from faster_whisper.audio import decode_audio
    audio = decode_audio(a.audio, sampling_rate=live.SR)
    chunks = cut_like_the_captioner(audio, a.chunk)
    seconds = sum(c[2] - c[1] for c in chunks)
    print(f"{len(chunks)} chunks, {seconds:.0f} s of speech would be uploaded to {a.provider} ({p['model']}).")
    if not a.yes:
        sys.exit("Nothing was sent. Add --yes to upload (check your university's recording rules first).")
    if not key and not os.environ.get("CLOUD_EVAL_BASE_URL"):
        sys.exit(f"Set {p['key']} in the environment.")
    prompt = ""
    if a.prompt_file:
        with open(os.path.expanduser(a.prompt_file), encoding="utf-8") as f:
            prompt = " ".join(f.read().split())
    terms = [t.strip() for t in re.split(r"[:,]", prompt)[1:] if t.strip()]

    texts, delays = [], []
    for k, (chunk, start, end, forced) in enumerate(chunks, 1):
        t0 = time.time()
        texts.append(transcribe_chunk(p, key, wav_bytes(chunk), prompt, terms).strip())
        delays.append(time.time() - t0)
        print(f"\r  {k}/{len(chunks)}", end="", flush=True)
    print()
    text = " ".join(texts)
    q = lambda x, f: sorted(x)[max(0, int(f * len(x)) - 1)]
    per_s = statistics.median([d / max(0.5, c[2] - c[1]) for d, c in zip(delays, chunks)])
    print(f"round trip per chunk: median {statistics.median(delays):.2f} s | p90 {q(delays, .9):.2f} s | max {max(delays):.2f} s"
          f"   (median {per_s:.2f} s per second of speech)")
    if a.ref:
        print(f"distance to the reference: {100 * distance(open(a.ref).read(), text):.1f} %")
    if a.out:
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
