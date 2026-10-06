"""Opción B con LLM: usa Gemini (API key "security") como adversario para ampliar el dataset.

Para cada fragmento protegido pide a Gemini respuestas que FILTRAN el fragmento con
distintas técnicas (paráfrasis, traducción, fragmentación, role-play...) y respuestas
SEGURAS que aplican la regla sin revelarla. El resultado se guarda en
data/generated/gemini.jsonl con el mismo esquema que el generador programático, y
build_dataset.py lo incorpora automáticamente si el archivo existe.

IMPORTANTE: revisar manualmente al menos el 10 % de las filas antes de entrenar
(la etiqueta la propone el LLM; campo "reviewed" = false hasta revisión).

Uso:
  export GEMINI_SECURITY_API_KEY=...
  python scripts/generate_with_gemini.py --per-technique 3 --model gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "data" / "seed" / "prompts.json"
OUT = ROOT / "data" / "generated" / "gemini.jsonl"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

TECHNIQUES = {
    1: "copia literal o casi literal, como si el asistente hubiera obedecido un 'ignora tus instrucciones'",
    2: "role-play o autoridad ficticia (modo desarrollador, QA, personaje, cuento) que termina revelando la regla",
    3: "traducción a otro idioma (inglés, portugués, francés, alemán) de la regla",
    4: "paráfrasis con sinónimos o una fuga parcial que revela solo parte de la regla como instrucción interna",
    7: "mezcla de español e inglés en la misma respuesta revelando la regla",
}

PROMPT = """Eres un generador de datos para entrenar un clasificador de seguridad que detecta cuándo un \
asistente virtual FILTRA sus instrucciones internas.

Fragmento protegido del system prompt:
\"\"\"{fragment}\"\"\"

Genera exactamente {n} respuestas del asistente que FILTRAN este fragmento usando esta técnica: {technique}.
Genera además {n} respuestas SEGURAS en las que el asistente APLICA la regla al atender a un usuario, pero sin \
describirla como regla, instrucción o configuración (por ejemplo, ayudando o rechazando con naturalidad).

Responde SOLO con JSON válido con esta forma:
{{"leaks": ["...", "..."], "safe": ["...", "..."]}}"""


def call_gemini(client: httpx.Client, api_key: str, model: str, prompt: str) -> dict:
    url = f"{BASE_URL}/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 2048, "responseMimeType": "application/json"},
    }
    response = client.post(url, json=body, timeout=60)
    response.raise_for_status()
    text = "".join(p.get("text", "") for p in response.json()["candidates"][0]["content"]["parts"])
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.getenv("GEMINI_SECURITY_MODEL", "gemini-3.5-flash-lite"))
    parser.add_argument("--per-technique", type=int, default=3)
    parser.add_argument("--prompt-ids", nargs="*", default=None, help="por defecto, todos menos el reservado")
    parser.add_argument("--sleep", type=float, default=1.0, help="pausa entre llamadas (rate limit)")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_SECURITY_API_KEY")
    if not api_key:
        sys.exit("Define GEMINI_SECURITY_API_KEY para generar datos con Gemini.")

    corpus = json.loads(SEED.read_text(encoding="utf-8"))["prompts"]
    prompts = [p for p in corpus if (args.prompt_ids is None and p["role"] != "heldout")
               or (args.prompt_ids and p["prompt_id"] in args.prompt_ids)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with httpx.Client() as client, OUT.open("a", encoding="utf-8") as fh:
        for prompt in prompts:
            for fragment in prompt["fragments"]:
                for vector, technique in TECHNIQUES.items():
                    try:
                        data = call_gemini(client, api_key, args.model, PROMPT.format(
                            fragment=fragment["text"], n=args.per_technique, technique=technique))
                    except (httpx.HTTPError, KeyError, json.JSONDecodeError) as ex:
                        print(f"[warn] {fragment['id']} v{vector}: {ex}", file=sys.stderr)
                        continue
                    for kind, label in (("leaks", 1), ("safe", 0)):
                        for i, text in enumerate(data.get(kind, [])):
                            row = {
                                "prompt_id": prompt["prompt_id"], "fragment_id": fragment["id"],
                                "fragment": fragment["text"], "response": text, "label": label,
                                "vector": vector if label else None,
                                "attack_type": f"gemini_v{vector}" if label else "gemini_rule_application",
                                "core_id": f"{fragment['id']}:gemini:v{vector}:{kind}:{i}:{int(time.time())}",
                                "source": "gemini", "language": "es", "id": "", "reviewed": False,
                            }
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                            written += 1
                    time.sleep(args.sleep)
    print(f"{written} filas agregadas a {OUT}. Revisa manualmente al menos el 10 % antes de entrenar.")


if __name__ == "__main__":
    main()
