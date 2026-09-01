# Komnae — build and deploy, start to finish

Five parts, in order. Don't skip ahead: Part 4 gives you a URL that Part 5
needs.

- **Part 1** — Backend running on your WSL machine (~20 min)
- **Part 2** — Frontend built in Lovable (~30 min)
- **Part 3** — Both repos on GitHub (~10 min)
- **Part 4** — Backend deployed, public URL (~15 min)
- **Part 5** — Frontend on Vercel, wired to the backend (~10 min)

---

## Part 1 — Backend on your machine

### 1.1 Unpack and set up

```bash
cd ~/dev
unzip ~/Downloads/komnae-backend.zip -d .
cd komnae-backend
```

If `python3 -m venv` fails, WSL is missing the venv package:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip build-essential
```

Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. If it doesn't, nothing below will
land in the right place.

### 1.2 Install — light first

`khmer-nlp-kcc` pulls in torch, which is a ~800 MB download. Skip it for local
development; you'll add it on the server in Part 4 where it belongs.

```bash
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" pydantic httpx pandas pyarrow huggingface_hub khmercut
```

Verify the segmenter works before going further:

```bash
python -c "from khmercut import tokenize; print(tokenize('ខ្ញុំទៅសាលារៀន'))"
```

You want `['ខ្ញុំ', 'ទៅ', 'សាលារៀន']`. If this errors, stop and fix it — every
layer above depends on segmentation.

### 1.3 Build the dictionary

```bash
python scripts/build_dict.py
```

This downloads the RAC dictionary from HuggingFace (~21 MB) and writes
`data/words.txt` and `data/entries.json`. It prints how many words it kept.
Expect roughly 30–40k headwords after filtering non-Khmer rows.

If you have your old 57k Chuon Nath list, drop it at `data/words_57k.txt`
before running this and the script will merge both, telling you how many words
each source contributed.

### 1.4 Get a Gemini key

Go to **aistudio.google.com/apikey**, sign in, create an API key. Free tier is
plenty for a competition demo.

```bash
cp .env.example .env
nano .env    # or: sed -i 's|GEMINI_API_KEY=|GEMINI_API_KEY=YOUR_KEY_HERE|' .env
```

`.env` is gitignored. Your key will not end up on GitHub.

### 1.5 Run it

```bash
export $(grep -v '^#' .env | xargs)
uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

You want `"segmenter": "khmercut"`, a `words` count in the tens of thousands,
and `"server_key": true`. If `segmenter` says `regex`, khmercut didn't load and
your results will be garbage — go back to 1.2.

Now an actual check:

```bash
curl -s -X POST localhost:8000/api/check \
  -H 'Content-Type: application/json' \
  -d '{"text":"ខ្ញុំទៅសាលោរៀនកាលពីស្អែក","use_ai":true}' \
  | python3 -m json.tool
```

Two things to look for. `"ai": "ok"` means Gemini answered. And the sentence
has a deliberate grammar error — ស្អែក is "tomorrow" but កាលពី means "in the
past" — so a well-behaved run flags it as `"type": "grammar"`. That's the layer
the dictionary can't do, and it's the thing to show judges.

**Interactive docs:** open `http://localhost:8000/docs` in Windows Chrome. FastAPI
generates a live API explorer. Useful for you, and genuinely impressive in a
demo when a judge asks "what's the backend doing?"

---

## Part 2 — Frontend in Lovable

Go to lovable.dev, start a new project, and paste this as your first message.

````
Build a Khmer writing assistant called Komnae — a single-page web editor.
Everything user-facing is in Khmer; code and comments in English.

LAYOUT
Full-height page. Top bar with "កំណែ Komnae" on the left and a settings gear
icon on the right. Center is a writing surface, max-width 760px, generous
padding, white card on a soft warm neutral background. This is the hero — it
should feel like a calm sheet of paper, not a form. A collapsible suggestions
panel sits on the right at 320px; on mobile it becomes a bottom sheet.

Bottom-left of the editor: live word count and a status pill that reads
"កំពុងពិនិត្យ..." while checking, "រកឃើញ ៣ កំហុស" when issues exist, or
"គ្មានកំហុស" when clean.

EDITOR
Use a contentEditable div, NOT a textarea — inline underlines have to render.
Khmer text at 20px with line-height 2.0, font-family "Noto Sans Khmer",
"Khmer OS Battambang", sans-serif. Khmer needs far more line spacing than
Latin; do not use a tight default.

Flagged spans get a wavy underline: red for type "spelling", blue for
"grammar", amber for "style". Clicking one opens a small popover anchored to
it containing: the suggestion as a large tappable button, the definition and
part of speech underneath it in smaller text, the reason in Khmer, up to three
alternatives as small chips, and an "មិនអើពើ" (ignore) link.

Accepting a suggestion replaces that span in place without collapsing the
selection or jumping the cursor to the end.

SUGGESTIONS PANEL
One card per issue: original struck through, arrow, suggestion in bold, short
reason below. Clicking a card scrolls to and highlights that span in the
editor. "ទទួលយកទាំងអស់" button at the top of the panel.

SETTINGS MODAL (gear icon)
- Password-type input for a personal Gemini API key, with a "សាកល្បង" test
  button that POSTs to /api/validate-key and shows a green check or red error.
- Text explaining the key is stored only in this browser and never saved on
  the server.
- A clear-key button.
- Toggle between "check as I type" and "check on button press".

API
Base URL comes from import.meta.env.VITE_API_URL. Two calls:

POST {API}/api/check
  headers: Content-Type: application/json, and X-Gemini-Key: <key> only if
  the user saved a personal key in settings
  body: { "text": string, "use_ai": boolean }
  returns: {
    issues: [{ start, end, original, suggestion, alternatives[], reason,
               type, source, definition, pos, confidence }],
    tokens: number,
    backend: string,
    ai: "ok" | "skipped" | "no_key" | "error" | "timeout",
    ai_error: string
  }

POST {API}/api/validate-key
  body: { "api_key": string }  returns: { valid, model, error }

CRITICAL: start and end are character offsets into the exact string that was
sent. Render every underline with text.slice(start, end) and nothing else. Do
NOT normalize, trim, or transform the text before sending it — that shifts
every offset and underlines land on the wrong letters.

BEHAVIOR
- Debounce checks to 800ms after typing stops.
- Send use_ai: true. The response always contains dictionary results even when
  the AI layer failed, so render issues regardless of the ai field.
- If ai is "error" or "timeout", still render the issues and show a quiet note
  "ការពិនិត្យដោយ AI មិនបានសម្រេច". If ai is "no_key", show a subtle prompt to
  add a key in settings. Never blank the editor on any error.
- Store the user's key in localStorage under "komnae_gemini_key".
- Empty state: a Khmer placeholder and one clickable example sentence.

STYLE
Warm, quiet, paper-like. One accent color. Rounded corners, soft shadows. No
purple gradients, no glassmorphism, no emoji in the UI.

For now mock both endpoints with setTimeout and fake data matching the shapes
above, so the whole UI is clickable before the backend is connected.
````

Iterate in Lovable until it looks right. The mock data means you can polish the
UI without the backend running.

### 2.1 Point it at your local backend

Once the UI works with mocks, tell Lovable:

```
Replace the mocked endpoints with real fetch calls to VITE_API_URL. Add a
.env file with VITE_API_URL=http://localhost:8000
```

Your backend already allows `localhost:5173`, `:3000` and `:8080` in CORS, so
Lovable's dev server can talk to it. If you see a CORS error in the browser
console, check which port Lovable is using and add it to `CORS_ORIGINS` in your
`.env`, then restart uvicorn.

---

## Part 3 — Both repos on GitHub

### 3.1 Frontend

Lovable has a GitHub button in the top right. Click it, authorize, and it
creates a repo and pushes for you. Nothing to do by hand.

### 3.2 Backend

```bash
cd ~/dev/komnae-backend
git init
git add .
git commit -m "Komnae backend: segmentation, RAC dictionary, Gemini layer"
```

Before pushing, confirm your key isn't in there:

```bash
git ls-files | grep -E '\.env$' && echo "STOP — .env is staged" || echo "safe: .env not tracked"
```

Create an empty repo on github.com (no README, no gitignore — you have both),
then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/komnae-backend.git
git branch -M main
git push -u origin main
```

---

## Part 4 — Deploy the backend

Vercel can't host this. `khmer-nlp-kcc` needs torch and transformers, which
unpack to roughly 800 MB against Vercel's 250 MB function limit. Use Hugging
Face Spaces — free, and the kcc model already lives on HF Hub so it caches into
the image.

### 4.1 Add the Spaces header

Spaces reads configuration from YAML frontmatter at the very top of README.md:

```bash
cd ~/dev/komnae-backend
cat > /tmp/hf-header.md <<'EOF'
---
title: Komnae API
emoji: ✍️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

EOF
cat /tmp/hf-header.md README.md > /tmp/readme-new.md && mv /tmp/readme-new.md README.md
head -10 README.md
```

`app_port: 7860` has to match the port in the Dockerfile. It does.

### 4.2 Enable the better segmenter

On the server you have room for torch, so turn it on:

```bash
sed -i 's/^# khmer-nlp-kcc/khmer-nlp-kcc/; s/^khmer-nlp-kcc$/khmer-nlp-kcc/' requirements.txt
grep khmer requirements.txt
```

Both `khmercut` and `khmer-nlp-kcc` should be listed. The segmenter prefers kcc
and falls back to khmercut automatically, so you get the better model with a
safety net.

### 4.3 Push to Spaces

Create a Space at huggingface.co/new-space. Name it `komnae-api`, SDK
**Docker**, hardware **CPU basic (free)**, visibility **Public**.

```bash
git add -A && git commit -m "Configure for HF Spaces"
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/komnae-api
git push space main
```

It'll ask for credentials — use your HF username and an access token from
huggingface.co/settings/tokens (write scope) as the password.

The first build takes 10–15 minutes because of torch. Watch the logs on the
Space page.

### 4.4 Add your secrets

On the Space page: **Settings → Variables and secrets**.

| Name | Type | Value |
| ---- | ---- | ----- |
| `GEMINI_API_KEY` | Secret | your key |
| `CORS_ORIGINS` | Variable | your Vercel URL (fill in after Part 5) |

The Space restarts automatically. Then:

```bash
curl -s https://YOUR_HF_USERNAME-komnae-api.hf.space/health | python3 -m json.tool
```

`"segmenter": "kcc"` means the transformer loaded. That's your public backend
URL — save it.

---

## Part 5 — Deploy the frontend

1. Go to vercel.com, **Add New → Project**, import your Lovable GitHub repo.
2. Vercel detects Vite automatically. Don't change the build settings.
3. Under **Environment Variables**, add:

   | Key | Value |
   | --- | ----- |
   | `VITE_API_URL` | `https://YOUR_HF_USERNAME-komnae-api.hf.space` |

   No trailing slash.
4. Deploy.

### 5.1 Close the CORS loop

Go back to your Space's settings and set `CORS_ORIGINS` to your Vercel URL,
e.g. `https://komnae.vercel.app`. Preview deployments are already covered by a
`*.vercel.app` regex in `main.py`, but your production domain should be
explicit.

Open your Vercel URL, type some Khmer with a deliberate mistake, and watch the
underline appear. That URL is what you give the judges.

---

## Before judging

**Warm the Space.** Free Spaces sleep after 48 hours idle and take ~30 seconds
to wake. Open the URL an hour before your demo. A judge clicking a link and
staring at a spinner is the worst possible first impression.

**Have a fallback.** Record a 60-second screen capture of the working app. If
the venue wifi dies, you still have something to show.

**Know your data provenance.** Judges ask. The answer: dictionary from
វចនានុក្រមខ្មែរ ២០២២ by the Royal Academy of Cambodia, segmentation from a
Khmer transformer model, verification by Gemini. Put that on an about page.

**Know what's yours.** The interesting engineering is the cluster-level edit
distance (Khmer typos change several code points at once, so ordinary edit
distance ranks bad candidates too highly), the offset preservation through
three layers, and the hallucination gates on the Gemini output. Those are worth
explaining out loud.

---

## When something breaks

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `"segmenter": "regex"` | khmercut/kcc didn't install | Reinstall; check `/health` |
| Underlines on wrong letters | Text normalized before sending | Send the raw string |
| CORS error in console | Origin not allowed | Add it to `CORS_ORIGINS`, restart |
| `"ai": "no_key"` | Key not set or not loaded | Check the Space secret |
| `"ai": "error"` | Gemini rejected or timed out | Read `ai_error` in the response |
| Every word flagged | Dictionary didn't build | Rerun `build_dict.py`, check `/health` |
| Space build fails on torch | Free tier ran out of disk | Drop kcc, use khmercut only |
