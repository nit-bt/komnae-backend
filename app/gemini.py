"""
The Gemini pass: verify the dictionary's guesses and catch what it cannot see.

Two jobs, one call:

  1. Judge each flagged word in context. The dictionary can only say "not in
     the wordlist" - it cannot tell a proper noun from a typo, or pick between
     five candidates that are all one edit away. The model can.
  2. Find grammar and word-choice problems the dictionary is blind to, since
     every word in "ខ្ញុំទៅផ្សារកាលពីស្អែក" is spelled correctly.

The model never invents replacements freely. Spelling corrections are gated
against the dictionary after the fact, and anything it returns that does not
appear verbatim in the source text is dropped. Those two rules are what keep a
confident-sounding hallucination out of the user's document.
"""

import json
import logging
import os

import httpx

from .dictionary import dictionary
from .models import Issue

log = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SERVER_KEY = os.environ.get("GEMINI_API_KEY", "")
TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "12"))

# How many flagged words to send. Long documents get truncated rather than
# producing a slow, expensive call the user will not wait for.
MAX_ISSUES = 40

SYSTEM_PROMPT = """\
You are a Khmer spelling assistant. Your only job is to decide, for each \
flagged word, whether it is misspelled and if so which candidate the writer \
meant.

You will receive the text, and a numbered list of words that a Khmer dictionary \
did not recognise, each with candidate corrections drawn from that dictionary.

Use the surrounding sentence to decide. A word is often one edit away from \
several real words, and only the context tells you which was intended: a \
candidate that fits the subject, tense and register of the sentence is the \
right one.

For each numbered word, decide one action:
  keep    - the word is fine. Proper nouns, names, place names, loanwords, \
brand names and modern coinages are fine even though the dictionary lacks them.
  replace - it is a misspelling. Choose the correct word FROM THE CANDIDATES \
LIST for that item. Do not invent a spelling that is not in its candidates.
  drop    - it is not a real error worth showing the user.

Do NOT rewrite the writer's wording. Do not change word order, tense, register \
or phrasing. Do not suggest a better way to say something. Do not report \
grammar or style problems. Correcting a spelling is the whole task: the writer \
chose their words and you are only fixing how they are spelled.

Write every `reason` in Khmer, in one short clause. Be conservative: a false \
alarm annoys the writer more than a missed error."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "action": {"type": "string", "enum": ["keep", "replace", "drop"]},
                    "chosen": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "action"],
            },
        },
    },
    "required": ["verdicts"],
}


class GeminiError(Exception):
    pass


class NoKeyError(GeminiError):
    pass


def _build_prompt(text: str, issues: list[Issue]) -> str:
    lines = ["TEXT:", text, "", "FLAGGED WORDS:"]

    if not issues:
        lines.append("(none - the dictionary recognised every word)")
    else:
        for i, issue in enumerate(issues):
            candidates = [c for c in ([issue.suggestion] + issue.alternatives) if c]
            described = []
            for c in candidates[:5]:
                gloss = dictionary.gloss(c)
                described.append(f"{c} ({gloss})" if gloss else c)
            shown = "; ".join(described) if described else "(no candidates found)"
            lines.append(f"{i}. \"{issue.original}\" -> candidates: {shown}")

    return "\n".join(lines)


async def refine(text: str, issues: list[Issue], api_key: str | None = None) -> list[Issue]:
    """
    Run the verification pass. Returns the corrected issue list.

    Raises NoKeyError when no key is configured, GeminiError on API failure.
    The caller is expected to fall back to the local issues on either.
    """
    key = api_key or SERVER_KEY
    if not key:
        raise NoKeyError("no Gemini API key configured")

    truncated = issues[:MAX_ISSUES]
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": _build_prompt(text, truncated)}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }

    url = f"{API_ROOT}/models/{MODEL}:generateContent"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            url,
            json=payload,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        )

    if response.status_code == 400 and "API key" in response.text:
        raise NoKeyError("the Gemini API key was rejected")
    if response.status_code != 200:
        raise GeminiError(f"Gemini returned {response.status_code}: {response.text[:200]}")

    try:
        body = response.json()
        raw = body["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw)
    except (KeyError, IndexError, ValueError) as exc:
        raise GeminiError(f"could not parse Gemini response: {exc}") from exc

    return _merge(text, truncated, issues[MAX_ISSUES:], parsed)


def _merge(text: str, judged: list[Issue], untouched: list[Issue], parsed: dict) -> list[Issue]:
    """Apply the model's verdicts, then append its grammar findings."""
    verdicts = {v["index"]: v for v in parsed.get("verdicts", []) if "index" in v}
    result: list[Issue] = []

    for i, issue in enumerate(judged):
        verdict = verdicts.get(i)
        if verdict is None:
            result.append(issue)
            continue

        action = verdict.get("action")
        if action in ("keep", "drop"):
            continue

        chosen = (verdict.get("chosen") or "").strip()

        # Hard gate: a spelling correction must be a real dictionary word, and
        # must have been one of the candidates we offered. Without this the
        # model will occasionally produce a plausible-looking non-word.
        offered = {issue.suggestion, *issue.alternatives} - {""}
        if not chosen or chosen not in offered or not dictionary.contains(chosen):
            log.debug("rejected ungrounded correction %r for %r", chosen, issue.original)
            result.append(issue)
            continue

        senses = dictionary.define(chosen)
        result.append(issue.model_copy(update={
            "suggestion": chosen,
            "alternatives": [a for a in offered if a != chosen],
            "reason": verdict.get("reason") or issue.reason,
            "source": "ai",
            "definition": senses[0].get("def", "") if senses else "",
            "pos": senses[0].get("pos", "") if senses else "",
            "confidence": 0.95,
        }))

    result.extend(untouched)

    result.sort(key=lambda i: i.start)
    return result


def _locate_additions(text: str, additions: list[dict], existing: list[Issue]) -> list[Issue]:
    """
    Turn the model's grammar findings into offset-anchored issues.

    Models are unreliable at reporting character positions, so we ignore any
    they give and search for the span ourselves. A span we cannot find verbatim
    is discarded: it means the model paraphrased, and we would be underlining
    text the user never wrote.
    """
    taken = [(i.start, i.end) for i in existing]
    found: list[Issue] = []
    cursor = 0

    for addition in additions:
        original = (addition.get("original") or "").strip()
        suggestion = (addition.get("suggestion") or "").strip()
        if not original or not suggestion or original == suggestion:
            continue

        start = text.find(original, cursor)
        if start == -1:
            start = text.find(original)
        if start == -1:
            log.debug("dropped ungrounded addition %r", original)
            continue

        end = start + len(original)
        if any(start < e and s < end for s, e in taken):
            continue

        taken.append((start, end))
        cursor = end
        found.append(Issue(
            start=start,
            end=end,
            original=original,
            suggestion=suggestion,
            reason=addition.get("reason", ""),
            type=addition.get("type", "grammar"),
            source="ai",
            confidence=0.8,
        ))

    return found


async def validate_key(api_key: str) -> tuple[bool, str]:
    """Cheap round-trip so the settings panel can verify a pasted key."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{API_ROOT}/models",
                headers={"x-goog-api-key": api_key},
            )
    except httpx.HTTPError as exc:
        return False, str(exc)

    if response.status_code == 200:
        return True, ""
    return False, f"{response.status_code}: {response.text[:120]}"
