# lecture-captions

Live English captions **with Chinese translation for classroom lectures, running entirely on your own laptop**.
Whisper ([faster-whisper](https://github.com/SYSTRAN/faster-whisper), int8, CPU) transcribes the lecture, an
offline model ([Argos Translate](https://github.com/argosopentech/argos-translate)) translates it, and a small local
web page shows the result. Built by a non-native listener for a MacBook in the front row.

[中文说明](README.zh-CN.md)

## What it does

- Listens to the microphone and shows **one complete sentence at a time** (English + Chinese), with the unfinished sentence in grey.
- **Local web page** with one-click buttons: pause (closes the microphone, ~0 % CPU), hide/show Chinese (layout does not jump), save to Markdown, font size.
- **Topic prompts** (`--prompt-file`): a short list of the lecture's vocabulary steers Whisper towards the right terms.
- **Silero VAD** finds pauses, so it works on quiet recordings and in noisy rooms (a plain loudness threshold did not).
- Saves the session audio and a running transcript (turn off with `--no-record`); press `s` for a Markdown file of the whole session.
- Works on a recording too: `./run.sh --file lecture.mp3`.

## Install

Needs Python 3.10+ and a microphone. Tested on an Intel MacBook Pro (macOS); Windows/Linux are untested.

```bash
git clone https://github.com/SyzenShen/lecture-captions.git
cd lecture-captions
./setup.sh          # venv, packages, offline en->zh model (~100 MB); Whisper small.en (~460 MB) downloads on first run
./run.sh --web --prompt-file prompts/example_biology.txt
```

macOS asks for microphone access for your terminal on first use.

## Use

| Key (terminal or web page) | |
|---|---|
| `s` | save the session so far as Markdown (`notes/`, or `CAPTIONS_NOTES_DIR`) |
| `c` | hide / show the Chinese |
| `p` | pause / resume (the microphone is really closed while paused) |

Useful options (`./run.sh --help` for all):

| Option | |
|---|---|
| `--web` | open the local page (`http://127.0.0.1:8765`, localhost only) |
| `--prompt-file FILE` / `--prompt "..."` | topic vocabulary, **English only**, keep it under ~500 characters |
| `--chunk 5` | lower latency (default 8 s: about 29 % less CPU, same accuracy in our tests) |
| `--translator none` | English only |
| `--device N`, `--list-devices` | pick the input device |
| `--vad energy` | the old loudness-based pause detection |

Copy `config.example.sh` to `config.local.sh` (git-ignored) to set where notes are saved.

### Writing a good prompt

The prompt is passed to Whisper as its opening context. Put the course name and 20–40 of the terms, abbreviations
and R/shell function names you expect, comma separated (see `prompts/example_biology.txt`).
In our own tests it was the single biggest improvement: on one real lecture, a prompt that matched the topic got 10 of 15
key terms right versus 6 with a prompt for a different course. It cuts both ways: a prompt for the wrong subject makes
Whisper insert its words into the captions, so change it when the subject changes.

## Honest limitations

- **English lectures only** (`small.en`). The model is small because it has to run in real time on a laptop CPU.
- **Accuracy depends heavily on accent, distance and noise.** There is no formal benchmark here. On noisy synthetic
  test speech the word error rate was roughly 13-18 %; real classroom audio with accented speakers and student questions
  is harder, and technical terms are where it fails most.
- The offline Chinese translation is basic and sometimes literal ("transcription" -> 抄写). It never sees the lecture
  context. `--translator claude` (needs `pip install anthropic` and `ANTHROPIC_API_KEY`) is better but **sends the English text to an API**.
- Captions lag the speaker by a few seconds.
- Program messages (terminal and web page buttons) are in Chinese; translations welcome.

## Privacy and recording

Audio is processed on your machine and never leaves it. The web page listens on `127.0.0.1` only.
Only `--translator claude` sends anything out (the English text, not the audio).
Recordings and transcripts are written to `transcripts/` and are git-ignored.
**Check your university's rules before recording a lecture**, and do not publish recordings of other people.

## Layout

```
live.py               the program (capture -> VAD -> Whisper -> translate -> terminal / web / Markdown)
run.sh  setup.sh      launcher and one-time setup
fetch_argos_zh.py     downloads the offline en->zh model without PyTorch
prompts/              example topic prompt
```
