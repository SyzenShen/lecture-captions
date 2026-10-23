"""Find the questions a lecturer asks, so the UI can highlight them and pin the latest ones.

Measured on a real 52-minute lecture: a third of the sentences end with "?", but more than half of those are
filler ("..., right?", "Yeah?", "So why?"). Only substantive questions are worth highlighting.
"""
import re

_SENTENCES = re.compile(r"(?<=[.?!])\s+")
_FILLER = re.compile(r"^(?:(?:so|and|but|now|okay|ok|well|then|alright|right|yeah|yes|um|uh)[,\s]+)+", re.I)
# "..., right?" "..., you know?" : a tag on a statement, not a question to the audience
_TAG = re.compile(r",\s*(?:right|yeah|okay|ok|no|correct|yes|see|you know|alright|isn't it|is it|don't you|doesn't it|aren't they|huh)\s*\?$", re.I)
_NEGATED = re.compile(r"^(?:do|does|did|can|could|would|will|should|is|are|was|were|have|has|had)(?:\s+not\b|n't\b)", re.I)

_AUX = r"(?:do|does|did|can|could|would|will|should|is|are|was|were|have|has|had|may|might)"
# question word order without a "?": an auxiliary followed by a subject ("Do you know...", "Are there...")
_INVERTED = re.compile(
    rf"^{_AUX}\s+(?:you|we|i|they|he|she|it|this|that|there|the|a|an|any(?:one|body)|some(?:one|body)|every(?:one|body)|these|those)\b",
    re.I)
# "How many ...", "What about ...", "What if ..."  ("who is ..." is NOT here: it is usually a relative clause)
_WH_PHRASE = re.compile(r"^(?:what|why|how|who|where|when|which)\s+(?:about|if|else|many|much|long|kind|type|exactly)\b", re.I)

MIN_WORDS = 3                  # after stripping "so", "and", ...; drops "Why?", "So why?", "Yeah?"
MAX_WORDS_WITHOUT_MARK = 18    # a long sentence starting with "how" is usually a statement
MAX_WORDS_SHOWN = 24           # long merged sentences ending in "?": show only the end


def is_question(sentence):
    """True if one sentence is (probably) a substantive question."""
    s = sentence.strip()
    if not s or _TAG.search(s):
        return False
    core = _FILLER.sub("", s).strip()
    words = len(core.split())
    if words < MIN_WORDS:
        return False
    if s.endswith("?"):
        return True
    if words > MAX_WORDS_WITHOUT_MARK or _NEGATED.match(core):
        return False
    return bool(_INVERTED.match(core) or _WH_PHRASE.match(core))


def extract_question(text):
    """The first question sentence inside `text` (it may hold several sentences), shortened for display, or ""."""
    for sentence in _SENTENCES.split(text.strip()):
        if is_question(sentence):
            words = sentence.split()
            if len(words) > MAX_WORDS_SHOWN:
                return "… " + " ".join(words[-MAX_WORDS_SHOWN:])
            return sentence.strip()
    return ""
