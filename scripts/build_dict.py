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

    print(f"\nwrote {words_path}    {len(words):,} words  "
          f"{os.path.getsize(words_path)/1e6:.1f} MB")
    print(f"wrote {entries_path}  {len(entries):,} entries  "
          f"{os.path.getsize(entries_path)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
