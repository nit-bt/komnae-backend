"""
The local pass: segment, then flag any token the dictionary does not know.

Runs in milliseconds with no network, so the frontend can underline
misspellings while the Gemini call is still in flight.

Two rules keep it from making confident wrong guesses, which matters because
this is what users see whenever the AI layer is unavailable:

  1. Compounds are not errors. Khmer builds words by joining words, but RAC
     lists only headwords. If an unknown word splits cleanly into two known
     words, it is almost certainly valid. This is what makes កាលពី (កាល + ពី)
     stop being flagged.

  2. A replacement is only offered when the dictionary is actually confident.
     A same-length substitution is a plausible typo. A candidate that is just
     a shorter piece of the input is not a correction, it is a different word.
     In that case the word is still flagged, but with no suggestion attached.
"""

from .dictionary import dictionary
from .khmer import is_khmer_word, split_kcc
from .models import Issue
from .segmenter import segmenter

# Single-cluster tokens are almost always particles, or fragments left behind
# by a bad segmentation. Flagging them produces more noise than signal.
MIN_LENGTH = 2

# Words shorter than this are not tested for compound structure: splitting a
# two-cluster word leaves halves too small for the test to mean anything.
MIN_COMPOUND_CLUSTERS = 3

REASON_UNKNOWN = "ពាក្យនេះមិនមាននៅក្នុងវចនានុក្រម"
REASON_MISSPELLED = "ពាក្យនេះមិនមាននៅក្នុងវចនានុក្រម សូមពិនិត្យអក្ខរាវិរុទ្ធ"


def is_compound(word: str) -> bool:
    """
    True if the word splits at a cluster boundary into two dictionary words.

    RAC lists headwords, not every legal combination, so compounds constantly
    look like unknown words. Checking both halves is cheap and removes a large
    class of false positives without any model call.
    """
    clusters = split_kcc(word)
    if len(clusters) < MIN_COMPOUND_CLUSTERS:
        return False

    for i in range(1, len(clusters)):
        left = "".join(clusters[:i])
        right = "".join(clusters[i:])
        if dictionary.contains(left) and dictionary.contains(right):
            return True

    return False


def is_confident(word: str, candidate: str) -> bool:
    """
    Whether a candidate is worth showing as a correction.

    Same cluster count means characters were swapped, which is what a typo
    looks like. A different count means clusters were added or removed, and
    when the candidate is simply contained in the input we are looking at a
    root word rather than a correction. កាល is not a fix for កាលពី.
    """
    if not candidate:
        return False

    if len(split_kcc(word)) != len(split_kcc(candidate)):
        return False

    return not (word.startswith(candidate) or word.endswith(candidate))


def check(text: str) -> tuple[list[Issue], int, str]:
    """Returns (issues, token count, segmenter backend)."""
    tokens = segmenter.segment(text)
    issues: list[Issue] = []

    for token in tokens:
        word = token.text

        if len(word) < MIN_LENGTH or not is_khmer_word(word):
            continue

        if dictionary.contains(word):
            continue

        # A valid compound of two known words is not an error.
        if is_compound(word):
            continue

        candidates = dictionary.candidates(word)
        best = candidates[0] if candidates else ""

        # Only propose a replacement the dictionary can stand behind. Without
        # this, a Gemini outage leaves users looking at wrong corrections
        # presented with the same confidence as right ones.
        if not is_confident(word, best):
            best = ""

        senses = dictionary.define(best) if best else []

        issues.append(Issue(
            start=token.start,
            end=token.end,
            original=word,
            suggestion=best,
            alternatives=[c for c in candidates[1:] if is_confident(word, c)],
            reason=REASON_MISSPELLED if best else REASON_UNKNOWN,
            type="spelling",
            source="dictionary",
            definition=senses[0].get("def", "") if senses else "",
            pos=senses[0].get("pos", "") if senses else "",
            confidence=0.9 if best else 0.4,
        ))

    boundaries = [(t.start, t.end) for t in tokens]
    return issues, len(tokens), segmenter.backend or "regex", boundaries
