"""
Dictionary: membership checks and spelling-candidate generation.

Loaded once at import time and reused across requests. Two assets:

    data/words.txt    the membership set (is this a real word?)
    data/entries.json word -> [{pos, pro, def}] for the suggestion popover

Build both with scripts/build_dict.py.
"""

import json
import logging
import os
from collections import defaultdict
from functools import lru_cache

from .khmer import normalize, split_kcc

log = logging.getLogger(__name__)

DATA_DIR = os.environ.get("KOMNAE_DATA_DIR", "data")
WORDS_PATH = os.path.join(DATA_DIR, "words.txt")
ENTRIES_PATH = os.path.join(DATA_DIR, "entries.json")

# Candidates are only compared against words whose cluster-length is within
# this much of the input. Cuts the search space by roughly 15x.
LENGTH_WINDOW = 2

# Above this cluster-edit-distance a "correction" is more likely to be a
# different word than a typo. Keeping this tight is what stops the suggestion
# list from turning into noise.
MAX_DISTANCE = 2


class Dictionary:
    def __init__(self, words_path: str = WORDS_PATH, entries_path: str = ENTRIES_PATH):
        self.words: set[str] = set()
        self.entries: dict[str, list[dict]] = {}
        # normalized form -> original surface form, so we can look up words
        # that are spelled correctly but stored in a different cluster order
        self.normalized: dict[str, str] = {}
        # cluster count -> list of (word, clusters)
        self.buckets: dict[int, list[tuple[str, list[str]]]] = defaultdict(list)

        self._load(words_path, entries_path)

    def _load(self, words_path: str, entries_path: str) -> None:
        if os.path.exists(words_path):
            with open(words_path, encoding="utf-8") as f:
                self.words = {line.strip() for line in f if line.strip()}
        else:
            log.warning("wordlist missing at %s, spellcheck disabled", words_path)

        if os.path.exists(entries_path):
            with open(entries_path, encoding="utf-8") as f:
                self.entries = json.load(f)
        else:
            log.warning("entries missing at %s, definitions disabled", entries_path)

        for word in self.words:
            clusters = split_kcc(word)
            self.buckets[len(clusters)].append((word, clusters))
            self.normalized.setdefault(normalize(word), word)

        log.info(
            "dictionary loaded: %d words, %d entries, %d buckets",
            len(self.words), len(self.entries), len(self.buckets),
        )

    # --- membership ----------------------------------------------------------

    def contains(self, word: str) -> bool:
        """True if the word is in the dictionary, allowing for cluster-order
        and split-vowel differences."""
        if word in self.words:
            return True
        return normalize(word) in self.normalized

    def canonical(self, word: str) -> str | None:
        """The dictionary's preferred spelling of a word, if it is known."""
        if word in self.words:
            return word
        return self.normalized.get(normalize(word))

    # --- entries -------------------------------------------------------------

    def define(self, word: str) -> list[dict]:
        """Sense entries for a word. Empty list if undefined."""
        canonical = self.canonical(word) or word
        return self.entries.get(canonical, [])

    def gloss(self, word: str) -> str:
        """A one-line definition, for the suggestion card and Gemini context."""
        senses = self.define(word)
        return senses[0].get("def", "") if senses else ""

    # --- candidates ----------------------------------------------------------

    @lru_cache(maxsize=8192)
    def candidates(self, word: str, limit: int = 5) -> tuple[str, ...]:
        """
        Nearest dictionary words by cluster-level edit distance.

        Distance is measured over character clusters rather than code points.
        In Khmer a single mistyped subscript changes several code points at
        once, so code-point distance ranks bad candidates too highly.
        """
        target = split_kcc(normalize(word))
        n = len(target)
        if not n:
            return ()

        scored: list[tuple[int, int, str]] = []

        for length in range(max(1, n - LENGTH_WINDOW), n + LENGTH_WINDOW + 1):
            for candidate, clusters in self.buckets.get(length, ()):
                distance = _bounded_edit_distance(target, clusters, MAX_DISTANCE)
                if distance is None:
                    continue
                # Prefer same first cluster: Khmer typos are far more often in
                # the middle or end of a word than in the opening consonant.
                prefix_penalty = 0 if clusters[0] == target[0] else 1
                scored.append((distance, prefix_penalty, candidate))

        scored.sort()
        return tuple(word for _, _, word in scored[:limit])


def _bounded_edit_distance(a: list[str], b: list[str], max_distance: int) -> int | None:
    """
    Levenshtein distance over cluster lists, abandoned once it exceeds
    max_distance. Returns None if the bound is exceeded.
    """
    if abs(len(a) - len(b)) > max_distance:
        return None

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]
        best_in_row = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(
                previous[j] + 1,       # deletion
                current[j - 1] + 1,    # insertion
                previous[j - 1] + cost # substitution
            )
            current.append(value)
            best_in_row = min(best_in_row, value)
        if best_in_row > max_distance:
            return None
        previous = current

    distance = previous[-1]
    return distance if distance <= max_distance else None


# Module-scope singleton. Import cost is paid once per worker process, which
# matters on hosts that keep the container warm between requests.
dictionary = Dictionary()
