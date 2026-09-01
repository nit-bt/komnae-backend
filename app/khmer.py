"""
Khmer text primitives: normalization and character-cluster (KCC) splitting.

Nothing in here changes string length in a way that would break offsets, with
one exception: `normalize()` may shorten text (split-vowel repair collapses two
code points into one). That is why normalization is only ever applied to
individual tokens for *comparison* purposes, never to the document text used
for offset math. See segmenter.py for how offsets are preserved.
"""

import re
import unicodedata

# --- Character classes -------------------------------------------------------

CONSONANT = "\u1780-\u17a2"
INDEP_VOWEL = "\u17a5-\u17b3"
COENG = "\u17d2"
DEP_VOWEL = "\u17b6-\u17c5"
SIGN = "\u17c6-\u17d3\u17dd"
DIGIT = "\u17e0-\u17e9"

# Zero-width and no-break characters. In Khmer, U+200B is a legitimate word
# separator, so it is treated as whitespace by the segmenter rather than being
# silently deleted from document text.
ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"

KHMER_CHAR = re.compile(f"[{CONSONANT}{INDEP_VOWEL}{COENG}{DEP_VOWEL}{SIGN}{DIGIT}]")
KHMER_WORD = re.compile(f"^[{CONSONANT}{INDEP_VOWEL}{COENG}{DEP_VOWEL}{SIGN}]+$")

# A Khmer character cluster: a base, any number of subscript (coeng) pairs,
# then dependent vowels, then diacritic signs. The trailing `|.` makes the
# pattern total, so splitting never drops characters.
_KCC = re.compile(
    f"(?:[{CONSONANT}]|[{INDEP_VOWEL}])"
    f"(?:{COENG}[{CONSONANT}{INDEP_VOWEL}])*"
    f"[{DEP_VOWEL}]*"
    f"[{SIGN}]*"
    "|."
)

# --- Normalization -----------------------------------------------------------

# Code points the Unicode standard deprecates for Khmer. Mapped to their
# modern equivalents, or removed where there is no equivalent.
DEPRECATED = {
    "\u17a3": "\u17a2",  # independent vowel QA -> consonant QA
    "\u17a4": "\u17a2\u17b6",  # QAA -> QA + AA
    "\u17b4": "",  # inherent vowel AQ, display-only
    "\u17b5": "",  # inherent vowel AA, display-only
    "\u17d3": "\u17c6",  # BATHAMASAT -> NIKAHIT
}

# Split vowels that some keyboards emit as two code points.
SPLIT_VOWELS = [
    ("\u17c1\u17b6", "\u17c4"),  # E + AA      -> OO
    ("\u17c1\u17b8", "\u17c4"),  # E + II      -> OO  (common mistype)
    ("\u17c1\u17bb", "\u17c5"),  # E + U       -> AU
]

_ZW_TABLE = dict.fromkeys(map(ord, ZERO_WIDTH), None)


def strip_invisible(text: str) -> str:
    """Remove zero-width characters. For dictionary keys, not document text."""
    return text.translate(_ZW_TABLE)


def _reorder_cluster(cluster: str) -> str:
    """
    Put a cluster into canonical order: base, coeng pairs, vowels, signs.

    Typing order and storage order diverge often enough in Khmer that two
    visually identical words can differ byte-for-byte. Without this, dictionary
    lookups miss on words that are spelled correctly.
    """
    if len(cluster) < 2:
        return cluster

    base = cluster[0]
    coengs, vowels, signs = [], [], []

    i = 1
    while i < len(cluster):
        ch = cluster[i]
        if ch == COENG and i + 1 < len(cluster):
            coengs.append(cluster[i:i + 2])
            i += 2
            continue
        if re.match(f"[{DEP_VOWEL}]", ch):
            vowels.append(ch)
        elif re.match(f"[{SIGN}]", ch):
            signs.append(ch)
        else:
            signs.append(ch)
        i += 1

    # Multiple coeng pairs keep their relative order; everything else sorts to
    # a stable canonical position.
    return base + "".join(coengs) + "".join(sorted(vowels)) + "".join(sorted(signs))


def normalize(text: str) -> str:
    """
    Canonicalize a Khmer string for comparison.

    Applies NFC, removes deprecated code points, repairs split vowels, and
    reorders each cluster. Safe to call on a single word; do not call on
    document text you still need offsets into.
    """
    if not text:
        return text

    text = unicodedata.normalize("NFC", text)
    text = strip_invisible(text)

    for old, new in DEPRECATED.items():
        if old in text:
            text = text.replace(old, new)

    for old, new in SPLIT_VOWELS:
        if old in text:
            text = text.replace(old, new)

    return "".join(_reorder_cluster(c) for c in split_kcc(text))


def split_kcc(text: str) -> list[str]:
    """Split into Khmer character clusters. Concatenating the result
    reproduces the input exactly."""
    return _KCC.findall(text)


def is_khmer_word(text: str) -> bool:
    """True if the string is entirely Khmer letters, vowels and signs."""
    return bool(text) and bool(KHMER_WORD.match(text))


def has_khmer(text: str) -> bool:
    return bool(KHMER_CHAR.search(text))
