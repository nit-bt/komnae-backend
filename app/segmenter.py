"""
Word segmentation with exact character offsets into the original text.

Three backends, tried in order:

    kcc       khmer-nlp-kcc, a transformer model. Best quality, needs torch.
    khmercut  a small Rust-backed segmenter. Good, fast, no torch.
    regex     Khmer-script runs only. A floor, not a real segmenter.

Whichever runs, the contract is the same: tokens carry (start, end) offsets
into the string that was passed in, unmodified. The frontend renders underlines
using those offsets, so an off-by-one here shows up as a squiggle under the
wrong letter.
"""

import logging
import os
import re
import threading
from dataclasses import dataclass

from .khmer import CONSONANT, DEP_VOWEL, INDEP_VOWEL, SIGN, COENG, has_khmer

log = logging.getLogger(__name__)

BACKEND = os.environ.get("KOMNAE_SEGMENTER", "auto")

_KHMER_RUN = re.compile(
    f"[{CONSONANT}{INDEP_VOWEL}{COENG}{DEP_VOWEL}{SIGN}]+"
)


@dataclass
class Token:
    text: str
    start: int
    end: int

    def as_dict(self) -> dict:
        return {"text": self.text, "start": self.start, "end": self.end}


class Segmenter:
    """Lazily loads a backend on first use and reuses it thereafter."""

    def __init__(self, preferred: str = BACKEND):
        self.preferred = preferred
        self.backend: str | None = None
        self._impl = None
        # khmercut and torch model loading are both racy on cold start. A
        # single lock around init avoids two requests initialising at once.
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._impl is not None or self.backend == "regex":
            return
        with self._lock:
            if self._impl is not None or self.backend == "regex":
                return
            order = (
                ["kcc", "khmercut", "regex"]
                if self.preferred == "auto"
                else [self.preferred, "regex"]
            )
            for name in order:
                try:
                    self._impl = self._load(name)
                except Exception as exc:  # noqa: BLE001 - any failure is a fallback
                    log.warning("segmenter %s unavailable: %s", name, exc)
                    continue
                self.backend = name
                log.info("segmenter backend: %s", name)
                return

    @staticmethod
    def _load(name: str):
        if name == "kcc":
            from khmer_nlp import KhmerNLP

            nlp = KhmerNLP()
            # Warm the model now rather than inside the first request.
            nlp.segment("ភាសាខ្មែរ")
            return nlp
        if name == "khmercut":
            from khmercut import tokenize

            tokenize("ភាសាខ្មែរ")
            return tokenize
        if name == "regex":
            return None
        raise ValueError(f"unknown segmenter: {name}")

    def segment(self, text: str) -> list[Token]:
        if not text or not has_khmer(text):
            return []

        self._ensure_loaded()

        try:
            if self.backend == "kcc":
                pieces = self._impl.segment(text).split()
            elif self.backend == "khmercut":
                pieces = [p for p in self._impl(text) if p.strip()]
            else:
                return self._regex_tokens(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("segmentation failed (%s), falling back to regex: %s",
                        self.backend, exc)
            return self._regex_tokens(text)

        aligned = align(text, pieces)
        if aligned is None:
            log.warning("alignment failed for %s backend, falling back", self.backend)
            return self._regex_tokens(text)
        return aligned

    @staticmethod
    def _regex_tokens(text: str) -> list[Token]:
        return [
            Token(m.group(), m.start(), m.end())
            for m in _KHMER_RUN.finditer(text)
        ]


def align(text: str, pieces: list[str]) -> list[Token] | None:
    """
    Map segmented pieces back onto the original text.

    Segmenters return content but drop whitespace, and some insert separators.
    This walks the original forward, skipping anything the segmenter dropped,
    and records true offsets. Returns None if a piece cannot be located, which
    means the backend rewrote characters and its output cannot be trusted for
    offset math.
    """
    tokens: list[Token] = []
    cursor = 0

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        index = text.find(piece, cursor)
        if index == -1:
            return None

        # A large jump means we skipped real content, not just whitespace.
        skipped = text[cursor:index]
        if skipped.strip() and has_khmer(skipped.strip()):
            log.debug("segmenter skipped Khmer content: %r", skipped)

        tokens.append(Token(piece, index, index + len(piece)))
        cursor = index + len(piece)

    return tokens


segmenter = Segmenter()
