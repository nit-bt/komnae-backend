---
title: Komnae API
emoji: ✍️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Komnae — backend

Khmer writing assistant API. Three layers, each catching what the one before it
cannot:

1. **Segmentation** — Khmer has no spaces between words, so nothing else can
   happen until the text is split. Uses `khmer-nlp-kcc` when available,
   `khmercut` otherwise.
2. **Dictionary** — the Royal Academy of Cambodia *Khmer Dictionary 2022*
   (44,706 entries). Flags unknown words and proposes corrections by
   cluster-level edit distance.
3. **Gemini** — judges each flagged word in context, picks among the
   dictionary's candidates, and catches grammar errors where every individual
   word is spelled correctly.

## Setup

```bash
git clone <your-repo> && cd komnae-backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_dict.py          # downloads RAC dictionary, writes data/
cp .env.example .env                  # add your Gemini key
uvicorn app.main:app --reload --port 8000
```

Then `curl localhost:8000/health` — it reports which segmenter loaded and how
many words are in the dictionary.

Without `khmer-nlp-kcc` or `khmercut` installed the segmenter falls back to a
regex that only finds Khmer-script runs. It will not split words. Check
`/health` if results look wrong.

## Endpoints

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET`  | `/health` | What loaded, which model, is a server key set |
| `POST` | `/api/check` | Full check: local pass, then Gemini |
| `POST` | `/api/refine` | Gemini pass over issues the client already has |
| `POST` | `/api/validate-key` | Test a user-supplied Gemini key |
| `GET`  | `/api/define/{word}` | Dictionary entry for one word |

### The response contract

```jsonc
{
  "issues": [{
    "start": 7,           // character offset into the text you sent
    "end": 14,            // exclusive
    "original": "សាលោរៀន",
    "suggestion": "សាលារៀន",
    "alternatives": ["សាលា"],
    "reason": "ពាក្យនេះមិនមាននៅក្នុងវចនានុក្រម",
    "type": "spelling",   // spelling | grammar | style
    "source": "ai",       // dictionary | ai
    "definition": "កន្លែងសម្រាប់បង្រៀននិងរៀន",
    "pos": "ន.",
    "confidence": 0.95
  }],
  "tokens": 7,
  "backend": "kcc",
  "ai": "ok"              // ok | skipped | no_key | error | timeout
}
```

`start` and `end` index the **exact string you sent**, before any
normalization. Render underlines with `text.slice(start, end)` and nothing
else. This is the one thing to get right — if you normalize the text on the
client before sending, every offset shifts and squiggles land on the wrong
letters.

`ai` tells you what happened to the Gemini pass. The local results are always
returned regardless, so `ai: "error"` still gives you a usable response. Never
blank the editor on it.

## Bring-your-own key

Send the user's key as an `X-Gemini-Key` header and it is used instead of the
server's for that request only. It is not logged or stored.

```js
fetch(`${API}/api/check`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-Gemini-Key": userKey },
  body: JSON.stringify({ text, use_ai: true }),
});
```

Keep the user's key in `localStorage` on the client. If you would rather it
never touch your server at all, call Gemini directly from the browser and use
`/api/check` with `use_ai: false` for the dictionary pass.

## Deploying

The frontend goes to Vercel. The backend cannot: `khmer-nlp-kcc` depends on
`torch` and `transformers`, which unpack to roughly 800 MB against Vercel's
250 MB function limit. Use a container host.

**Hugging Face Spaces** is the closest fit, since the kcc checkpoint already
lives on HF Hub and gets cached in the image:

1. Create a Space, SDK **Docker**, hardware **CPU basic** (free).
2. `git push` this repo to the Space remote.
3. Settings → Variables and secrets → add `GEMINI_API_KEY` and `CORS_ORIGINS`.
4. Your URL is `https://<user>-<space>.hf.space`.

Render, Railway and Fly all work too — same Dockerfile, change the port.

If you would rather stay on Vercel for everything, drop `khmer-nlp-kcc` from
`requirements.txt` and set `KOMNAE_SEGMENTER=khmercut`. The whole thing then
fits in a Vercel Python function. You lose POS tagging and the transformer's
better handling of compound words.

Free Spaces sleep after 48 hours idle and take ~30s to wake. Before judging,
open the URL once to warm it.

### Connecting the frontend

In Vercel, set `VITE_API_URL` to your backend URL. In the backend, set
`CORS_ORIGINS` to your Vercel domain. Preview deployments are already covered
by a `*.vercel.app` regex in `main.py`.

## Data

Dictionary data is from **វចនានុក្រមខ្មែរ ២០២២, រាជបណ្ឌិត្យសភាកម្ពុជា**
(Royal Academy of Cambodia), via the `seanghay/khmer-dictionary-44k` dataset.
Marked research use only, non-commercial — fine for a competition entry, but
credit it on your about page.

## Tuning

Both knobs are in `dictionary.py`:

- `MAX_DISTANCE` (default 2) — how far a candidate may be from the typed word.
  Raise it and you catch more errors but suggest more nonsense.
- `LENGTH_WINDOW` (default 2) — how many clusters of length difference to
  search. This is the performance knob; it is what keeps lookups fast.
