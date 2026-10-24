"""Course glossary: correct known recognition mistakes and keep technical terms right in the Chinese translation.

A glossary is a plain text file, one rule per line (UTF-8, '#' starts a comment):

    calcium indicator line = 钙离子指示剂品系       # translate this term the way I want
    ~ multi-world plate = multi-well plate          # Whisper keeps hearing this wrongly: fix the English itself

Terms match case-insensitively, a space also matches a hyphen, and a plural "s"/"es" is accepted.
"""
import re

TAG = "Zorvax"     # a made-up proper noun: the offline translator copies it through unchanged, so it can be swapped back


def _pattern(term):
    words = re.split(r"[\s\-]+", term.strip())
    body = r"[\s\-]+".join(re.escape(w) for w in words if w)
    return re.compile(r"(?<![\w])" + body + r"(e?s)?(?![\w])", re.I)


class Glossary:
    def __init__(self, lines=()):
        self.fixes = []        # (pattern, replacement English)
        self.terms = []        # (pattern, English term, Chinese)
        for line in lines:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            left, right = (x.strip() for x in line.split("=", 1))
            if not left or not right:
                continue
            if left.startswith("~"):
                self.fixes.append((_pattern(left[1:]), right))
            else:
                self.terms.append((_pattern(left), left, right))
        self.terms.sort(key=lambda t: -len(t[1]))      # longest first: "calcium indicator line" before "line"
        self.seen = set()                              # terms already shown with their English once

    @classmethod
    def load(cls, paths):
        lines = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                lines += f.read().splitlines()
        return cls(lines)

    def __bool__(self):
        return bool(self.fixes or self.terms)

    def fix_english(self, text):
        """Replace known recognition mistakes in the English text."""
        for pat, right in self.fixes:
            text = pat.sub(lambda m, r=right: r + (m.group(1) or ""), text)
        return text

    def protect(self, text):
        """Swap each glossary term for a placeholder before translation. Returns (text, {placeholder: (English as spoken, Chinese)})."""
        tags, spans = {}, []
        for pat, _, zh in self.terms:
            for m in pat.finditer(text):
                if all(m.end() <= a or m.start() >= b for a, b, _ in spans):
                    spans.append((m.start(), m.end(), zh))
        for k, (a, b, zh) in enumerate(sorted(spans)):
            if k >= 26:
                break
            tags[f"{TAG}{chr(65 + k)}"] = (text[a:b], zh)
        out, last = [], 0
        for k, (a, b, zh) in enumerate(sorted(spans)[:26]):
            out += [text[last:a], f"{TAG}{chr(65 + k)}"]
            last = b
        out.append(text[last:])
        return "".join(out), tags

    def restore(self, zh, tags):
        """Put the Chinese terms back. The first time a term is used in a session its English follows in brackets.
        Returns None when the translator lost a placeholder (the caller then translates without the glossary)."""
        for tag, (english, chinese) in tags.items():
            pat = re.compile(re.escape(TAG) + r"\s?" + re.escape(tag[len(TAG):]), re.I)
            if not pat.search(zh):
                return None
        for tag, (english, chinese) in tags.items():
            pat = re.compile(re.escape(TAG) + r"\s?" + re.escape(tag[len(TAG):]), re.I)
            shown = chinese
            if chinese not in self.seen:
                self.seen.add(chinese)
                shown = f"{chinese}（{english.lower()}）"
            zh = pat.sub(lambda _m, s=shown: s, zh, count=1)
            zh = pat.sub(chinese, zh)
        return zh


class GlossaryTranslator:
    """Wraps another translator so the glossary terms come out right."""

    def __init__(self, base, glossary):
        self.base, self.glossary = base, glossary

    def __call__(self, text):
        protected, tags = self.glossary.protect(text)
        if not tags:
            return self.base(text)
        zh = self.glossary.restore(self.base(protected), tags)
        return zh if zh is not None else self.base(text)
