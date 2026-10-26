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
- **Questions are highlighted** (accent border) and the **three most recent are pinned** in a bar at the top, so you can see what is being discussed when the room starts talking. The Markdown notes end with a list of every question asked.
- **Connectives in bold** (`b` toggles): but, because, so, therefore, if, for example ... are printed bold in the English, so the structure of a long sentence is easier to follow ("and"/"or" are left out: they are everywhere). The Chinese line is in regular weight so its connectives (但是, 因为, 所以, 如果 ...) stand out in bold too.
- **Focus mode** (`f`): only the last three sentences, larger; click a sentence to open its Chinese. Read less, listen more; the full transcript is saved for after class.
- Works on a recording too: `./run.sh --file lecture.mp3`.

## Install

Needs Python 3.10+ and a microphone. Tested on an Intel MacBook Pro (macOS); Windows/Linux are untested.

```bash
git clone https://github.com/SyzenShen/lecture-captions.git
cd lecture-captions
./setup.sh          # venv, packages, offline en->zh model (~100 MB); Whisper small.en (~460 MB) downloads on first run
./run.sh example
```

macOS asks for microphone access for your terminal on first use.

## Use

Put one prompt file per course (or per lesson) in `prompts/` and, if you like, a glossary in `glossary/`; the first word of the file name is the course name
(`prompts/ngs_full.txt`, `prompts/ngs_3_rnaseq.txt`, `glossary/ngs.txt` all belong to course `ngs`).

```bash
./run.sh ngs 3      # course ngs, lesson 3 (the lesson is optional; without it ngs_full is used)
./run.sh ngs        # the course's own prompt + glossary
./run.sh            # lists the courses; Enter repeats the last one
```

Names can be shortened to a unique start (`./run.sh mol`). The page opens in your browser, the profile follows the power source
(battery: `battery`; plugged in: `balanced`; `CAPTIONS_PROFILE=plugged` in `config.local.sh` or `--profile plugged` changes that),
and start-up prints five lines: course, profile, page address, keys, where the notes go. `./run.sh --list-courses` shows what was found.
A shell alias makes it shorter still: `alias cap='~/lecture-captions/run.sh'`.

| Key (terminal or web page) | |
|---|---|
| `s` | save the session so far as Markdown (`notes/`, or `CAPTIONS_NOTES_DIR`) |
| `c` | hide / show the Chinese |
| `p` | pause / resume (the microphone is really closed while paused) |
| `b` | connectives in bold on / off |
| `f` | focus mode: last three sentences only, Chinese on click (`c` then shows/hides all of them) |

Useful options (`./run.sh --help` for all):

| Option | |
|---|---|
| `--no-web` | terminal only; by default the local page (`http://127.0.0.1:8765`, localhost only) opens |
| `--prompt-file FILE` / `--prompt "..."` (instead of a course name) | topic vocabulary, **English only**, keep it under ~500 characters |
| `--profile battery\|balanced\|plugged` | power situation (see below); default `balanced` |
| `--glossary FILE` | course glossary: right Chinese for terms, fixes for mishearings (see below) |
| `--no-draft` | no grey draft text: about 40 % less CPU on battery, first words appear later |
| `--chunk 8` | longest piece per recognition (default 6 s; 8 s uses less CPU, 5 s is quicker) |
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

## Power profiles

Measured on a 7-minute real stretch of lecture (a fast, accented lecturer; 2019 Intel MacBook Pro, 6 cores), whole product
running in real time, no translator; one core = 100 %. Accuracy is the word distance to a Whisper large-v3 decode of the whole stretch
(its own noise floor against turbo is ~13 %). "Before" is the version of 2026-10-24 morning measured in the same way.

| Profile | What runs | CPU before | CPU now | Accuracy before / now | First words | Accurate text replaces the draft after |
|---|---|---|---|---|---|---|
| `battery` | `small.en` on merged ~6 s pieces, beam 3, 1 thread, no draft | 122 % | **52 %** | 25.7 % / 23.6 % | - | about 12 s |
| `balanced` (default) | + `tiny.en` grey drafts | 179 % | **87 %** | 25.7 % / 18.9 % | median 1.6 s (p90 3.5 s) | median 12.7 s (p90 18 s) |
| `plugged` | + `large-v3-turbo` re-reads every ~15-27 s block (it ends at a pause) and replaces its sentences | about 410 % (not re-measured with the savings above) | - | 17.8-21.6 % (3 stretches, measured before the savings) | as `balanced` | blocks ~10 s after they end |

```bash
./run.sh --profile plugged my_course
```

Where the savings come from: the speech is decoded in merged pieces ending at a pause (`--merge-seconds`) instead of one decode per
3.5 s piece (each decode pads its audio to 15 s, so short pieces waste most of the work, and longer context reads better);
OpenMP threads wait passively instead of spinning; beam 3 and one thread cost no measurable accuracy on blocks this long;
the draft model's first draft comes after 1 s of speech and later drafts are spaced 1.5 s. The price is that the accurate text arrives
later than before (6 s); `--merge-seconds 0` restores the old timing at about twice the CPU. The Chinese follows the accurate text.

`--refine-model`, `--refine-threads`, `--block-seconds` and `--beam` set the pieces separately. If the refiner falls more than three
blocks behind, the oldest block is left as it was. The first run of `plugged` downloads `large-v3-turbo` (~1.6 GB).

The grey drafts are rough on purpose: they differ from the accurate text in about 40-47 % of the words, so read them as
"what is being said right now", not as notes. Refining a block with turbo needs a temperature fallback (the program does this);
without it turbo looped on one lecture and was 35 % off instead of 19.5 %.

## Glossary: fewer misleading translations

The offline translator knows little about technical terms and sees one sentence at a time: "galvanometric mirror system" came out as
"camera mirror system", "calcium indicator lines" as "calcium index lines". And when Whisper mishears ("multi-world plates" for
"multi-well plates"), the translation turns the mistake into fluent Chinese. A glossary fixes both for a course:

```
calcium indicator line = 钙离子指示剂品系     # translate this term the way I want
~ multi-world plate = multi-well plate      # Whisper keeps hearing this wrongly: correct the English
```

```bash
./run.sh my_course                  # picks glossary/my_*.txt automatically; or --glossary FILE (may be given more than once)
```

Terms are swapped for a placeholder before translation and back afterwards, and shown as the English you hear with the Chinese in brackets,
in dark red on the web page: multi-well plates（多孔板）. The English is what the lecturer says, so a wrong guess of the translator
cannot mislead you (`--brackets first`: English with Chinese only the first time, plain Chinese after that).
A term that is hard to translate can be written `galvanometric mirror = -`: the sentence then shows just the English.
If the translator drops a placeholder, that sentence is translated without the glossary.
Glossary files are re-read whenever they change, so you can add a term in the break and it applies to the next sentence. Glossaries named `glossary/common_*.txt` are used for every course (basic biochemistry, say). See `glossary/example_biology.txt`; your own glossaries in `glossary/` are git-ignored. A larger offline model did not help: NLLB-200 (600M and 1.3B)
dropped characters, stopped in the middle of long sentences and used 7-12x the CPU of the current translator.

## After class: accurate notes

```bash
./refine.py transcripts/2026-10-23_1051.wav --prompt-file prompts/my_course.txt
```

Decodes the whole saved recording at once with a stronger model (default `large-v3-turbo`, ~1.6 GB download) and writes
`<name>.refined.md` with times, Chinese and the list of questions. On a real 7-minute stretch of lecture the English was
13.4 % from a Whisper large-v3 reference (the same as whole-file turbo, at the noise floor of that comparison: turbo and
large-v3 differ by 12.9 % from each other), versus 25.7 % for the live output. It takes about 0.8x the length of the audio
on the test machine, so run it while the laptop is charging. Bigger models matter less than you would think on poor audio:
on the same recording `medium.en` was no better than `small.en`.

## Honest limitations

- **English lectures only** (`small.en`). The model is small because it has to run in real time on a laptop CPU.
- **Accuracy depends heavily on accent, distance and noise.** There is no formal benchmark here. On noisy synthetic
  test speech the word error rate was roughly 13-18 %; real classroom audio with accented speakers and student questions
  is harder, and technical terms are where it fails most.
- The offline Chinese translation is basic and sometimes literal ("transcription" -> 抄写). It never sees the lecture
  context. `--translator claude` (needs `pip install anthropic` and `ANTHROPIC_API_KEY`) is better but **sends the English text to an API**.
- Delay, measured on a real 7-minute stretch of lecture (a fast, accented lecturer, 2019 Intel MacBook Pro): the first words of a
  segment appear in grey a median **1.6 s** after they were spoken (p90 3.5 s) and are replaced by the accurate text after a median of
  12.7 s (`balanced`; see the power profiles for the trade-off). The first version took 9.9 s to show anything.
- The live text is less accurate than it could be: it listens in short pieces to stay quick. Use `refine.py` after class.
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
