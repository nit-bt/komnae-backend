"""
The local pass: segment, then flag any token the dictionary does not know.

This runs in a few milliseconds and needs no network, so the frontend can
underline misspellings while the Gemini call is still in flight.
"""

from .dictionary import dictionary
from .khmer import is_khmer_word, normalize
from .models import Issue
from .segmenter import segmenter

# Single-cluster tokens are almost always particles or fragments of a word the
# segmenter split badly. Flagging them produces more noise than signal.
MIN_LENGTH = 2


def check(text: str) -> tuple[list[Issue], int, str]:
    """Returns (issues, token count, segmenter backend)."""
    tokens = segmenter.segment(text)
    issues: list[Issue] = []

    for token in tokens:
        word = token.text

        if len(word) < MIN_LENGTH or not is_khmer_word(word):
            continue

        if dictionary.contains(word):
            # Known word, but possibly stored in a non-canonical cluster order.
            # Silently correcting that keeps text clean without nagging.
            canonical = dictionary.canonical(word)
            if canonical and canonical != word and normalize(word) == normalize(canonical):
                continue
            continue

        candidates = dictionary.candidates(word)
        best = candidates[0] if candidates else ""
        senses = dictionary.define(best) if best else []

        issues.append(Issue(
            start=token.start,
            end=token.end,
            original=word,
            suggestion=best,
            alternatives=list(candidates[1:]),
            reason="ពាក្យនេះមិនមាននៅក្នុងវចនានុក្រម" if not best
                   else "ពាក្យនេះមិនមាននៅក្នុងវចនានុក្រម សូមពិនិត្យអក្ខរាវិរុទ្ធ",
            type="spelling",
            source="dictionary",
            definition=senses[0].get("def", "") if senses else "",
            pos=senses[0].get("pos", "") if senses else "",
            confidence=0.9 if best else 0.5,
        ))

    return issues, len(tokens), segmenter.backend or "regex"
