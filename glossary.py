"""Course glossary: correct known recognition mistakes and keep technical terms right in the Chinese translation.

A glossary is a plain text file, one rule per line (UTF-8, '#' starts a comment):

    calcium indicator line = 钙离子指示剂品系       # translate this term the way I want
    galvanometric mirror = -                        # hard to translate: show the English in brackets instead
    multi-well plate = 多孔板 !                      # always show the English in brackets after the Chinese
    ~ multi-world plate = multi-well plate          # Whisper keeps hearing this wrongly: fix the English itself

Without "!" the English follows the Chinese only the first time the term appears in a session.
Terms match case-insensitively, a space also matches a hyphen, and a plural "s"/"es" is accepted.
"""
import os
import re
import threading

MARK_S, MARK_E = "\ue000", "\ue001"      # around a glossary term in the translation: the web page prints it in colour
TAG = "Zorvax"     # a made-up proper noun: the offline translator copies it through unchanged, so it can be swapped back


def strip_marks(text):
    """The translation as plain text (terminal, files, notes): without the markers that colour terms on the web page."""
    return text.replace(MARK_S, "").replace(MARK_E, "")


def _pattern(term):
    words = re.split(r"[\s\-]+", term.strip())
    body = r"[\s\-]+".join(re.escape(w) for w in words if w)
    return re.compile(r"(?<![\w])" + body + r"(e?s)?(?![\w])", re.I)


class Glossary:
    def __init__(self, lines=(), always=False, paths=()):
        self.always = always   # show the English in brackets every time, not just the first
        self.paths = list(paths)
        self.mtimes = self._stamp()
        self.fixes = []        # (pattern, replacement English)
        self.terms = []        # (pattern, English term, Chinese or None = keep the English, always bracket)
        self.seen = set()      # terms already shown with their English once (for always=False)
        self.lock = threading.Lock()
        self._parse(lines)

    def _parse(self, lines):
        fixes, terms = [], []
        for line in lines:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            left, right = (x.strip() for x in line.split("=", 1))
            if not left or not right:
                continue
            if left.startswith("~"):
                fixes.append((_pattern(left[1:]), right))
            else:
                bang = right.endswith("!")
                right = right.rstrip("!").strip()
                terms.append((_pattern(left), left, None if right in ("-", "") else right, bang))
        terms.sort(key=lambda t: -len(t[1]))           # longest first: "calcium indicator line" before "line"
        self.fixes, self.terms = fixes, terms

    def _stamp(self):
        out = []
        for p in self.paths:
            try:
                out.append(os.stat(p).st_mtime)
            except OSError:
                out.append(None)
        return out

    @staticmethod
    def _read(paths):
        lines = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                lines += f.read().splitlines()
        return lines

    def refresh(self):
        """Pick up edits to the glossary files while the program runs (add a term in the break, it applies at once)."""
        if not self.paths:
            return
        stamp = self._stamp()
        if stamp == self.mtimes:
            return
        with self.lock:
            self.mtimes = stamp
            try:
                self._parse(self._read(self.paths))
            except (OSError, UnicodeDecodeError):
                pass                                   # a half-saved file: keep the old rules until the next change

    @classmethod
    def load(cls, paths, always=False):
        return cls(cls._read(paths), always, paths)

    def __bool__(self):
        return bool(self.fixes or self.terms)

    def fix_english(self, text):
        """Replace known recognition mistakes in the English text."""
        self.refresh()
        for pat, right in self.fixes:
            text = pat.sub(lambda m, r=right: r + (m.group(1) or ""), text)
        return text

    def protect(self, text):
        """Swap each glossary term for a placeholder before translation.
        Returns (text, {placeholder: (English as spoken, Chinese or None, always bracket)})."""
        self.refresh()
        tags, spans = {}, []
        for pat, _, zh, bang in self.terms:
            for m in pat.finditer(text):
                if all(m.end() <= a or m.start() >= b for a, b, _, _ in spans):
                    spans.append((m.start(), m.end(), zh, bang))
        spans = sorted(spans)[:26]
        for k, (a, b, zh, bang) in enumerate(spans):
            tags[f"{TAG}{chr(65 + k)}"] = (text[a:b], zh, bang)
        out, last = [], 0
        for k, (a, b, zh, bang) in enumerate(spans):
            # a placeholder glued to punctuation ("ZorvaxA.") is broken up by the translator's tokenizer: leave a space
            gap = " " if text[b:b + 1] in (".", ",", ";", ":", "!", "?") else ""
            out += [text[last:a], f"{TAG}{chr(65 + k)}", gap]
            last = b
        out.append(text[last:])
        return "".join(out), tags

    def restore(self, zh, tags):
        """Put the Chinese terms back. The first time a term is used in a session its English follows in brackets.
        Returns None when the translator lost a placeholder (the caller then translates without the glossary)."""
        for tag in tags:
            if not re.search(re.escape(TAG) + r"\s?" + re.escape(tag[len(TAG):]), zh, re.I):
                return None
        for tag, (english, chinese, bang) in tags.items():
            pat = re.compile(re.escape(TAG) + r"\s?" + re.escape(tag[len(TAG):]), re.I)
            if chinese is None:                          # hard to translate: the English in brackets is the term
                shown = f"（{english}）"
            elif bang or self.always or chinese not in self.seen:
                self.seen.add(chinese)
                shown = f"{chinese}（{english.lower()}）"
            else:
                shown = chinese
            shown = MARK_S + shown + MARK_E
            zh = pat.sub(lambda _m, s=shown: s, zh, count=1)
            zh = pat.sub(lambda _m, s=shown: s, zh)
        return re.sub(r"\s+([。，！？；：])", r"\1", zh)


class GlossaryTranslator:
    """Wraps another translator so the glossary terms come out right."""

    def __init__(self, base, glossary):
        self.base, self.glossary = base, glossary

    def __call__(self, text):
        protected, tags = self.glossary.protect(text)
        if not tags:
            return self.base(text)
        if re.fullmatch(r"\W*" + TAG + r"A\W*", protected):      # the sentence is just the term: nothing to translate
            zh = self.glossary.restore(protected.strip(" .,;:!?") + "。", tags)
            return zh
        zh = self.glossary.restore(self.base(protected), tags)
        return zh if zh is not None else self.base(text)
