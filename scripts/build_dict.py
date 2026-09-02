#!/usr/bin/env python3
"""
Build Komnae's dictionary assets from the RAC Khmer Dictionary 2022 dataset.

Outputs:
    data/words.txt    - one word per line, the membership set (union with your
                        existing Chuon Nath list if present)
    data/entries.json - word -> [{pos, pro, def}], for the suggestion popover

Run:  python build_dict.py
"""

import glob
import json
import os
import re
import unicodedata
from collections import defaultdict

import pandas as pd
from huggingface_hub import snapshot_download

REPO = "seanghay/khmer-dictionary-44k"
FREQ_REPO = "https://github.com/sbbic/khmerlbdict.git"
FREQ_FILES = ["KHSV.txt", "KHOV.txt", "seafreq.txt"]
SYMSPELL_REPO = "https://github.com/sungkhum/tiptap-khmer-line-breaker.git"
OUT_DIR = "data"

# Your existing 57k Chuon Nath list. Adjust if it lives elsewhere.
EXISTING_WORDLIST = "data/words_57k.txt"

MAX_SENSES = 2       # senses kept per word, keeps the JSON small
MAX_DEF_LEN = 160    # characters; the popover shows one line anyway

# Invisible characters that poison exact-match lookups.
INVISIBLES = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00a0"), None)

KHMER_RE = re.compile(r"^[\u1780-\u17ff\u19e0-\u19ff]+$")


def clean(text):
    """Strip invisibles and NFC-normalize. Returns '' for non-strings."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    return text.translate(INVISIBLES).strip()


def build_symspell(words: set) -> dict:
    """
    A 76,847-word list with corpus frequencies, MIT licensed.

    RAC lists headwords, which leaves real words missing: inflected forms,
    compounds and proper nouns all get flagged as errors. This roughly doubles
    coverage and carries better frequency data than the SBBIC corpus, which
    matters because frequency is what breaks ties between candidates that are
    equally far from what was typed.

    Also returns variant spelling pairs, where both forms are correct and
    neither should be flagged.
    """
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp()
    try:
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", SYMSPELL_REPO, tmp + "/tk"],
            check=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  could not fetch symspell data ({exc}); using RAC only")
        return {}

    base = os.path.join(tmp, "tk", "public", "dictionaries")
    freq = {}

    path = os.path.join(base, "km_symspell_dictionary.txt")
    if os.path.exists(path):
        added = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) >= 2 and parts[1].isdigit():
                    word = clean(parts[0])
                    if word and KHMER_RE.match(word):
                        if word not in words:
                            added += 1
                        words.add(word)
                        freq[word] = int(parts[1])
        print(f"  symspell added {added:,} words RAC did not have")

    # Variant spellings: both sides are correct, so both belong in the wordlist.
    path = os.path.join(base, "khmer-multiple-spellings.txt")
    if os.path.exists(path):
        variants = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                for part in line.split("="):
                    word = clean(part)
                    if word and KHMER_RE.match(word) and word not in words:
                        words.add(word)
                        variants += 1
        print(f"  variant spellings added {variants:,} words")

    return freq


def build_frequencies():
    """
    Corpus word frequencies from the SBBIC line-breaking dictionary.

    Two candidates one edit away are not equally likely corrections: people
    mistype toward words they actually write. គ្រូ appears 2,543 times in this
    corpus and ក្រក not at all, which is what stops the ranking falling back
    to alphabetical order among tied candidates.
    """
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp()
    try:
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", FREQ_REPO, tmp + "/lb"],
            check=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  could not fetch frequencies ({exc}); ranking will be weaker")
        return

    freq = {}
    for name in FREQ_FILES:
        path = os.path.join(tmp, "lb", "src", name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) >= 2 and parts[1].strip().isdigit():
                    word = parts[0].strip()
                    if word:
                        freq[word] = max(freq.get(word, 0), int(parts[1]))

    out = os.path.join(OUT_DIR, "freq.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(freq, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {out}  {len(freq):,} words with frequency")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"downloading {REPO} ...")
    path = snapshot_download(repo_id=REPO, repo_type="dataset")
    files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
    if not files:
        raise SystemExit("no parquet files found in the snapshot")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {len(df):,} rows, {df.word.nunique():,} unique words")

    entries = defaultdict(list)
    skipped = 0

    for row in df.itertuples(index=False):
        word = clean(row.word)

        # Drop entries that aren't pure Khmer script (single letters, Latin,
        # digits). They are dictionary artifacts, not spellcheckable words.
        if not word or not KHMER_RE.match(word):
            skipped += 1
            continue

        if len(entries[word]) >= MAX_SENSES:
            continue

        definition = clean(row.definition)
        if len(definition) > MAX_DEF_LEN:
            definition = definition[:MAX_DEF_LEN].rstrip() + "…"

        entries[word].append({
            "pos": clean(row.pos),
            "pro": clean(row.pro).strip("[]"),
            "def": definition,
        })

    print(f"  kept {len(entries):,} headwords, skipped {skipped:,} non-Khmer rows")

    # Membership set = RAC headwords, unioned with your existing list.
    words = set(entries)
    if os.path.exists(EXISTING_WORDLIST):
        with open(EXISTING_WORDLIST, encoding="utf-8") as f:
            existing = {clean(line) for line in f}
        existing = {w for w in existing if w and KHMER_RE.match(w)}
        new_from_rac = words - existing
        print(f"  existing list: {len(existing):,} words")
        print(f"  RAC adds {len(new_from_rac):,} words your list didn't have")
        words |= existing
    else:
        print(f"  note: {EXISTING_WORDLIST} not found, using RAC headwords only")

    words_path = os.path.join(OUT_DIR, "words.txt")
    with open(words_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(words)))

    entries_path = os.path.join(OUT_DIR, "entries.json")
    with open(entries_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, separators=(",", ":"))

    extra_freq = build_symspell(words)

    build_frequencies()

    # Merge the richer symspell counts over the SBBIC ones.
    freq_path = os.path.join(OUT_DIR, "freq.json")
    merged = {}
    if os.path.exists(freq_path):
        with open(freq_path, encoding="utf-8") as f:
            merged = json.load(f)
    merged.update(extra_freq)
    with open(freq_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  merged frequencies: {len(merged):,} words")

    # words.txt was written before the symspell list was merged in, so rewrite
    # it now that the set is complete.
    with open(words_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(words)))

    print(f"\nwrote {words_path}    {len(words):,} words  "
          f"{os.path.getsize(words_path)/1e6:.1f} MB")
    print(f"wrote {entries_path}  {len(entries):,} entries  "
          f"{os.path.getsize(entries_path)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
