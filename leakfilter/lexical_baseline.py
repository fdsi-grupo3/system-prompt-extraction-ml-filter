"""Port fiel a Python del Output Filter léxico actual (OutputFilterService.java).

Se usa SOLO como línea base en la evaluación, para comparar vector por vector el
filtro léxico (solapamiento de palabras + Levenshtein, umbral 0.45) contra el
clasificador entrenado sobre exactamente el mismo conjunto de respuestas.
"""

from __future__ import annotations

import base64
import binascii
import re

from rapidfuzz.distance import Levenshtein

STOPWORDS = {
    "de", "la", "el", "los", "las", "en", "que", "con", "por", "para", "un", "una", "unos", "unas",
    "y", "o", "ni", "no", "se", "su", "sus", "al", "es", "del", "lo", "tu", "tus", "te", "me", "le",
    "les", "esta", "este", "estos", "estas", "muy", "mas", "pero", "sin", "sobre", "como", "cuando",
}
MIN_MEANINGFUL_TOKENS = 3
BASE64_CANDIDATE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


def _normalize(text: str) -> str:
    text = re.sub(r"[^a-záéíóúñ0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> set[str]:
    return {w for w in _normalize(text).split() if len(w) > 2 and w not in STOPWORDS}


def _token_overlap(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    min_size = min(len(ta), len(tb))
    if min_size < MIN_MEANINGFUL_TOKENS:
        return 0.0
    return len(ta & tb) / min_size


def _levenshtein_similarity(a: str, b: str) -> float:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return 1.0 - Levenshtein.distance(a, b) / max_len


def _similarity(a: str, b: str) -> float:
    return max(_token_overlap(a, b), _levenshtein_similarity(_normalize(a), _normalize(b)))


def _decode_embedded_base64(text: str) -> str:
    decoded = []
    for match in BASE64_CANDIDATE.finditer(text):
        try:
            # Java usa Base64.getDecoder() estricto: sin relleno automático.
            raw = base64.b64decode(match.group(), validate=True).decode("utf-8", errors="replace")
        except (binascii.Error, ValueError):
            continue
        printable = sum(1 for c in raw if 32 <= ord(c) < 127 or c.isalpha())
        if raw.strip() and printable >= len(raw) * 0.85:
            decoded.append(raw)
    return " ".join(decoded).strip()


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def lexical_score(response: str, fragments: list[str], history: list[str] | None = None) -> float:
    decoded = _decode_embedded_base64(response)
    text = response if not decoded else f"{response} {decoded}"
    best = 0.0
    for fragment in fragments:
        best = max(best, _similarity(text, fragment))
        for sentence in _split_sentences(text):
            best = max(best, _similarity(sentence, fragment))
    if history:
        accumulated = " ".join(history[-5:]) + " " + text
        for fragment in fragments:
            best = max(best, _similarity(accumulated, fragment))
    return best
