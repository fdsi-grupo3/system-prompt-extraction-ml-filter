"""Preprocesamiento determinista compartido por entrenamiento, evaluación y servicio.

El Transformer solo entiende texto en lenguaje natural: no decodifica Base64 ni
ROT13, y los homoglifos/caracteres invisibles le cambian la tokenización. Por eso,
antes de llamar al modelo, la respuesta del LLM se expande en varias "vistas"
(original normalizada + versiones decodificadas) y cada vista se parte en
ventanas que caben en la longitud máxima del modelo. El veredicto final es el
máximo de P(fuga) sobre todas las combinaciones (vista × ventana × fragmento).
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from dataclasses import dataclass

# Caracteres invisibles usados para partir palabras (Vector 6).
_INVISIBLE = re.compile("[​‌‍⁠­﻿]")

# Homoglifos cirílicos/griegos → latino (mismo mapa que InputGuardService, ampliado).
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "і": "i", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "ѕ": "s", "ԛ": "q", "ԑ": "e", "ⅰ": "i", "А": "A", "В": "B", "Е": "E", "К": "K",
    "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "ο": "o",
    "α": "a", "ε": "e", "ι": "i", "ν": "v", "ρ": "p", "τ": "t", "Ο": "O", "Α": "A",
})

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_LEET_WORD = re.compile(r"\b(?=\w*[a-záéíóúñ])(?=\w*[013457@$])[\w@$]{3,}\b", re.IGNORECASE)

_BASE64_CANDIDATE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_HEX_CANDIDATE = re.compile(r"\b(?:[0-9a-fA-F]{2}[\s:]?){12,}\b")

# Palabras muy frecuentes en ES/EN/PT/FR/IT, para decidir si un texto "parece" lenguaje natural.
_COMMON_WORDS = frozenset(
    """de la que el en y a los se del las un por con no una su para es al lo como mas pero sus le
    ya o este si porque esta entre cuando muy sin sobre tambien me hasta hay donde quien desde todo
    nos durante todos uno les ni contra otros ese eso ante ellos e esto mi antes algunos que unos yo
    otro otras otra tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo
    nosotros tu tus debes puedes nunca solo reglas instrucciones
    the of and to in is you that it he was for on are as with his they at be this have from or one
    had by not but what all were we when your can said there use an each which she do how their if
    will up other about out many then them these so some her would make like him into time has look
    never only must should rules instructions
    o da em um para com uma os no se na por mais as dos como mas ao ele das seu sua ou quando muito
    le les des est et pas pour dans ce qui ne sur se plus par je avec tout faire son il elle
    il di che e la per un non sono una gli del della mai solo""".split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")

# Límite aproximado de caracteres por ventana de respuesta: deja espacio al fragmento
# protegido dentro de los 256 tokens del modelo (~4 caracteres por token en español).
DEFAULT_WINDOW_CHARS = 600
MAX_WINDOWS_PER_VIEW = 6


@dataclass(frozen=True)
class View:
    """Una forma de leer la respuesta (original, decodificada, acumulada…)."""

    name: str
    text: str


def normalize(text: str) -> str:
    """NFKC + sin caracteres invisibles + homoglifos a latino. No cambia mayúsculas ni acentos."""
    text = _INVISIBLE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_HOMOGLYPHS)
    return re.sub(r"[ \t]+", " ", text).strip()


def natural_language_ratio(text: str) -> float:
    words = re.findall(r"[a-záéíóúñçàèìòùâêôãõ]+", text.lower())
    if not words:
        return 0.0
    return sum(w in _COMMON_WORDS for w in words) / len(words)


def _is_printable_text(text: str) -> bool:
    if len(text.strip()) < 8:
        return False
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    return printable >= 0.95 * len(text) and letters >= 0.5 * len(text.replace(" ", ""))


def decode_base64_segments(text: str) -> list[tuple[str, str]]:
    """Devuelve (segmento original, texto decodificado) por cada substring Base64 legible."""
    decoded = []
    for match in _BASE64_CANDIDATE.finditer(text):
        candidate = match.group()
        padded = candidate + "=" * (-len(candidate) % 4)
        try:
            raw = base64.b64decode(padded, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if _is_printable_text(raw):
            decoded.append((candidate, raw))
    return decoded


def decode_hex_segments(text: str) -> list[tuple[str, str]]:
    """Devuelve (segmento original, texto decodificado) por cada secuencia hexadecimal legible."""
    decoded = []
    for match in _HEX_CANDIDATE.finditer(text):
        digits = re.sub(r"[\s:]", "", match.group())
        try:
            raw = bytes.fromhex(digits).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if _is_printable_text(raw):
            decoded.append((match.group(), raw))
    return decoded


def rot13_view(text: str) -> str | None:
    """Devuelve ROT13(text) solo si el resultado parece mucho más lenguaje natural que el original."""
    rotated = codecs.decode(text, "rot13")
    original_ratio = natural_language_ratio(text)
    rotated_ratio = natural_language_ratio(rotated)
    if rotated_ratio >= 0.25 and rotated_ratio > original_ratio + 0.15:
        return rotated
    return None


def reversed_view(text: str) -> str | None:
    """Texto escrito al revés ("senoiccurtsni sim"): se invierte si así parece más natural."""
    flipped = text[::-1]
    if natural_language_ratio(flipped) >= 0.25 and natural_language_ratio(flipped) > natural_language_ratio(text) + 0.15:
        return flipped
    return None


def leet_view(text: str) -> str | None:
    if len(_LEET_WORD.findall(text)) < 2:
        return None
    return _LEET_WORD.sub(lambda m: m.group().translate(_LEET), text)


def build_views(response: str, history: list[str] | None = None) -> list[View]:
    """Expande una respuesta en todas las vistas que el clasificador debe evaluar."""
    clean = normalize(response)
    decoded_views = []
    original = clean
    for name, segments in (("base64", decode_base64_segments(clean)), ("hex", decode_hex_segments(clean))):
        for i, (segment, decoded) in enumerate(segments):
            decoded_views.append(View(f"{name}[{i}]", normalize(decoded)))
            # El segmento ya se evalúa decodificado: en la vista original solo sumaría cientos de
            # tokens sin significado (latencia) y diluiría el resto de la respuesta.
            original = original.replace(segment, f"[{name}]")
    views = [View("original", original), *decoded_views]
    for name, transform in (("rot13", rot13_view), ("reversed", reversed_view), ("leet", leet_view)):
        transformed = transform(clean)
        if transformed:
            views.append(View(name, transformed))

    if history:
        # Vector 4 (reconstrucción incremental): las respuestas previas de la sesión más la
        # actual se evalúan juntas, para detectar fugas repartidas en varios turnos.
        accumulated = " ".join(normalize(h) for h in history[-4:]) + " " + clean
        views.append(View("accumulated", accumulated))
    return views


def split_windows(text: str, max_chars: int = DEFAULT_WINDOW_CHARS,
                  max_windows: int = MAX_WINDOWS_PER_VIEW) -> list[str]:
    """Agrupa oraciones en ventanas de hasta max_chars, con una oración de solapamiento.

    El solapamiento evita que una fuga que cae justo en el límite entre dos ventanas
    quede partida y diluida en ambas.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    windows: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        while len(sentence) > max_chars:  # oración gigante: se corta a la fuerza
            windows.append(sentence[:max_chars])
            sentence = sentence[max_chars - 100:]
        if current and len(" ".join(current + [sentence])) > max_chars:
            windows.append(" ".join(current))
            current = [current[-1]] if len(current[-1]) + len(sentence) < max_chars else []
        current.append(sentence)
    if current:
        windows.append(" ".join(current))

    if len(windows) > max_windows:
        # Se conservan las ventanas más recientes: en la vista acumulada el final es lo nuevo.
        windows = windows[-max_windows:]
    return windows
