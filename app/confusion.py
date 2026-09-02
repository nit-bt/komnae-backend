"""
Which Khmer characters get mistaken for which.

Ordinary edit distance treats every substitution as equally likely, which is
wrong for Khmer. The alphabet is organised into two series, and each first
series consonant has a second series partner that is its near twin: ក/គ, ច/ជ,
ត/ទ, ប/ព and so on. Writers confuse those constantly, because they sound close
and the choice depends on the vowel that follows.

Without this, ក្រូ produces ក្រក, ក្រង and ក្រប before គ្រូ. All four are one
substitution away, so they tie, and the tie breaks alphabetically. Weighting the
series pairs puts the plausible correction first.
"""

# Consonant series pairs. Each first series consonant beside its second series
# counterpart, plus a few sets that share a shape or a sound closely enough to
# be swapped in practice.
_CONFUSABLE_GROUPS = [
    "កគ",      # velar stops
    "ខឃ",
    "ចជ",      # palatal stops
    "ឆឈ",
    "ដទត",     # dental and alveolar stops
    "ឋឌឍថធ",
    "ណន",      # dental nasals, a very common swap
    "បព",      # labial stops
    "ផភ",
    "មម",
    "យ\u1799",
    "រ\u179a",
    "លឡ",      # the retroflex l is often typed as plain l
    "វ\u179c",
    "សឝឞ",     # sibilants
    "ហ\u17a0",
    "អ\u17a2",
]

# Vowels that render similarly or differ by a single stroke.
_CONFUSABLE_GROUPS += [
    "\u17b6\u17b7",        # AA / I
    "\u17b7\u17b8",        # I / II
    "\u17b9\u17ba",        # Y / YY
    "\u17bb\u17bc",        # U / UU
    "\u17c1\u17c2\u17c3",  # E / AE / AI
    "\u17c4\u17c5",        # OO / AU
]

# character -> set of characters it is plausibly confused with
CONFUSABLE: dict[str, set[str]] = {}
for group in _CONFUSABLE_GROUPS:
    for char in group:
        CONFUSABLE.setdefault(char, set()).update(set(group) - {char})


def confusion_penalty(typed: str, candidate: str) -> int:
    """
    How implausible this correction is as a typo, lower being more plausible.

    Only meaningful for equal-length strings differing in one position, which
    is the case this exists to disambiguate. Returns 0 when the differing
    characters are a known confusable pair, 1 otherwise.
    """
    if len(typed) != len(candidate):
        return 1

    differences = [
        (a, b) for a, b in zip(typed, candidate) if a != b
    ]

    if len(differences) != 1:
        return 1

    a, b = differences[0]
    return 0 if b in CONFUSABLE.get(a, ()) else 1
