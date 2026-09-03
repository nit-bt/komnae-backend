"""
Read Khmer text from a photograph.

Uses Tesseract's Khmer model rather than a vision model. It runs in the
container with no API key or quota, and it cannot invent text: where a language
model fills an unreadable gap with a plausible word, Tesseract produces
something visibly wrong or nothing at all. That matters here, because whatever
comes out goes straight to a spellchecker, and a hallucinated word is
indistinguishable from a typo once it reaches the editor.

The accuracy comes from `read_image`. Tesseract's output varies a lot with page
segmentation mode and preprocessing, and there is no way to know in advance
which combination suits a given photograph. So it runs several and scores each
result against the dictionary: the read containing the most real Khmer words
wins. On a hard image that is the difference between ខ្ញុំទៅសាលារៀនកាលពីស្អែក and
១ សខ-ខទៅសាលារៀនកាលពស្អេក — two plausible-looking strings, only one of which is
made of words.
"""

import io
import logging
import re
import shutil
import subprocess
import tempfile

log = logging.getLogger(__name__)

# 6 assumes a uniform block, 7 and 11 a single line or sparse text, 3 full
# automatic. Photographs of a paragraph and of one sentence want different
# ones, and users will send both.
PSM_MODES = ("6", "7", "11", "3")

# Punctuation Tesseract invents at the start of a line when it mistakes a page
# edge or shadow for a character.
LEADING_JUNK = re.compile(r"^[\s._\-|~`'\"]+")


class OcrUnavailable(Exception):
    """Tesseract or its Khmer model is not installed."""


def available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        langs = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:  # noqa: BLE001
        return False
    return "khm" in langs.split()


def _preprocess(data: bytes) -> bytes:
    """
    Clean up a photograph so Tesseract can read it.

    Phone photos are dimmer, noisier and lower contrast than the scans
    Tesseract was trained on. Returns the bytes unchanged if the imaging
    libraries are missing, since a worse image beats no image.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        return data

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Median rather than Gaussian: Khmer's subscript consonants are thin
        # enough that edge softening loses them, and a lost coeng turns
        # ខ្ញុំ into ខំ.
        array = cv2.medianBlur(array, 3)

        if array.shape[0] < 1000:
            array = cv2.resize(array, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Adaptive, because a photograph is unevenly lit and a single global
        # threshold loses whichever half is darker.
        array = cv2.adaptiveThreshold(
            array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )

        ok, encoded = cv2.imencode(".png", array)
        return encoded.tobytes() if ok else data
    except Exception as exc:  # noqa: BLE001
        log.debug("preprocessing failed, using original: %s", exc)
        return data


def _run(data: bytes, psm: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(data)
        f.flush()
        result = subprocess.run(
            ["tesseract", f.name, "-", "-l", "khm", "--psm", psm],
            capture_output=True, text=True, timeout=60,
        )
    return result.stdout.strip()


def _score(text: str) -> float:
    """
    How much of this read is actually Khmer words.

    Length is a poor proxy for quality: a bad read is often longer than a good
    one, because it turns page artefacts and shadows into characters.
    Segmenting and checking against the dictionary measures the thing that
    actually matters.
    """
    if not text.strip():
        return 0.0

    try:
        from .dictionary import dictionary
        from .segmenter import segmenter
    except Exception:  # noqa: BLE001
        return len(text) / 1000.0

    tokens = [t.text for t in segmenter.segment(text) if t.text.strip()]
    if not tokens:
        return 0.0

    known = sum(1 for t in tokens if dictionary.contains(t))

    # Ratio first, with a small length term so a tie goes to the longer read.
    # Without it, a single recognised word would beat a whole sentence.
    return known / len(tokens) + min(len(tokens), 60) / 10_000.0


def _tidy(text: str) -> str:
    lines = [LEADING_JUNK.sub("", line).rstrip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def read_image(data: bytes) -> str:
    """
    Extract Khmer text from image bytes.

    Runs Tesseract across preprocessed and original input at several
    segmentation modes, then keeps whichever read contains the most real words.
    """
    if not available():
        raise OcrUnavailable("tesseract with the Khmer model is not installed")

    best, best_score = "", -1.0

    for source in (_preprocess(data), data):
        for psm in PSM_MODES:
            try:
                text = _tidy(_run(source, psm))
            except subprocess.TimeoutExpired:
                log.warning("tesseract timed out at psm %s", psm)
                continue
            except Exception as exc:  # noqa: BLE001
                log.warning("tesseract failed at psm %s: %s", psm, exc)
                continue

            if not text:
                continue

            score = _score(text)
            if score > best_score:
                best, best_score = text, score

    log.info("ocr kept a read scoring %.2f", best_score)
    return best
