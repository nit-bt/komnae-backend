"""
Komnae backend.

    GET  /health          liveness plus what actually loaded
    POST /api/check       local spellcheck, then optionally the Gemini pass
    POST /api/refine      Gemini pass over issues the client already has
    POST /api/validate-key test a user-supplied Gemini key
    GET  /api/define/{w}  dictionary entry for one word

The frontend may send its own key in an `X-Gemini-Key` header, which takes
precedence over the server's. That key is used for the upstream call and is
never logged or stored.
"""

import logging
import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import gemini
from .checker import check
from .dictionary import dictionary
from . import extract as extraction
from .models import (
    CheckRequest, CheckResponse, ExtractRequest, ExtractResponse, Issue, Token,
    KeyCheckRequest, KeyCheckResponse, RefineRequest,
)
from .segmenter import segmenter

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("komnae")

app = FastAPI(title="Komnae API", version="1.0.0")

# Your Vercel domains. Set CORS_ORIGINS as a comma-separated list in prod;
# the default covers local Lovable/Vite development.
DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:3000,http://localhost:8080"
origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    # Vercel preview deployments get a new subdomain per push, so match the
    # whole vercel.app space rather than re-deploying on every branch.
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Gemini-Key"],
)


@app.on_event("startup")
async def warm_up() -> None:
    """Load the segmenter at boot so the first real request is not the one
    that pays for a torch model load."""
    segmenter.segment("ភាសាខ្មែរជាភាសាជាតិ")
    log.info("ready: backend=%s words=%d", segmenter.backend, len(dictionary.words))


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "segmenter": segmenter.backend or "not loaded",
        "words": len(dictionary.words),
        "entries": len(dictionary.entries),
        "gemini_model": gemini.MODEL,
        "server_key": bool(gemini.SERVER_KEY),
    }


@app.post("/api/check", response_model=CheckResponse)
async def check_text(
    request: CheckRequest,
    x_gemini_key: str | None = Header(default=None),
) -> CheckResponse:
    issues, token_count, backend, boundaries = check(request.text)
    marks = [Token(start=a, end=b) for a, b in boundaries]

    if not request.use_ai:
        return CheckResponse(issues=issues, tokens=token_count, backend=backend,
                             boundaries=marks, ai="skipped")

    try:
        refined = await gemini.refine(request.text, issues, x_gemini_key)
        return CheckResponse(issues=refined, tokens=token_count, backend=backend,
                             boundaries=marks, ai="ok")
    except gemini.NoKeyError as exc:
        return CheckResponse(issues=issues, tokens=token_count, backend=backend,
                             boundaries=marks, ai="no_key", ai_error=str(exc))
    except Exception as exc:  # noqa: BLE001
        # The local results are still good. Never fail the whole request
        # because the AI layer had a bad moment.
        log.warning("gemini pass failed: %s", exc)
        return CheckResponse(issues=issues, tokens=token_count, backend=backend,
                             boundaries=marks, ai="error",
                             ai_error=f"{type(exc).__name__}: {exc}"[:200])


@app.post("/api/refine", response_model=CheckResponse)
async def refine_issues(
    request: RefineRequest,
    x_gemini_key: str | None = Header(default=None),
) -> CheckResponse:
    """For clients that render local results first and upgrade them after."""
    try:
        refined = await gemini.refine(request.text, request.issues, x_gemini_key)
        return CheckResponse(issues=refined, tokens=len(request.issues),
                             backend=segmenter.backend or "", ai="ok")
    except gemini.NoKeyError as exc:
        return CheckResponse(issues=request.issues, tokens=len(request.issues),
                             backend=segmenter.backend or "", ai="no_key", ai_error=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.warning("gemini refine failed: %s", exc)
        return CheckResponse(issues=request.issues, tokens=len(request.issues),
                             backend=segmenter.backend or "", ai="error", ai_error=f"{type(exc).__name__}: {exc}"[:200])


@app.post("/api/validate-key", response_model=KeyCheckResponse)
async def validate_key(request: KeyCheckRequest) -> KeyCheckResponse:
    valid, error = await gemini.validate_key(request.api_key)
    return KeyCheckResponse(valid=valid, model=gemini.MODEL if valid else "", error=error)


@app.get("/api/define/{word}")
async def define(word: str) -> dict:
    senses = dictionary.define(word)
    if not senses:
        raise HTTPException(status_code=404, detail="word not found")
    return {
        "word": dictionary.canonical(word) or word,
        "senses": senses,
        "source": "វចនានុក្រមខ្មែរ ២០២២, រាជបណ្ឌិត្យសភាកម្ពុជា",
    }


@app.post("/api/extract", response_model=ExtractResponse)
async def extract_document(request: ExtractRequest) -> ExtractResponse:
    """Pull text out of an uploaded document so the editor can check it."""
    try:
        payload = extraction.decode_payload(request.data)
        text, note = extraction.extract(payload, request.mime_type, request.filename)
    except extraction.ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("unexpected extraction failure")
        raise HTTPException(status_code=500, detail="មិនអាចអានឯកសារនេះបានទេ") from exc

    return ExtractResponse(text=text, note=note, characters=len(text))


@app.get("/api/suggest/{prefix}")
async def suggest(prefix: str, limit: int = 10) -> dict:
    """
    Words beginning with the given prefix, for search-as-you-type.

    Exact lookup is the wrong shape for a search box: a user typing សា has not
    made a mistake, they are partway through a word. This returns headwords to
    choose from, each with a short gloss so the list is scannable.
    """
    prefix = prefix.strip()
    if len(prefix) < 1:
        return {"prefix": prefix, "matches": []}

    matches = sorted(w for w in dictionary.words if w.startswith(prefix))

    # Shorter words first: they are the more likely intended headword, and
    # longer compounds push the useful results off the end of the list.
    matches.sort(key=len)

    return {
        "prefix": prefix,
        "matches": [
            {"word": w, "gloss": dictionary.gloss(w)[:80]}
            for w in matches[:limit]
        ],
        "total": len(matches),
    }
