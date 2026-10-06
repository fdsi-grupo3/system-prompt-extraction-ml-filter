# FDSI-GP-03 · Output Filter de IA

Clasificador propio, independiente de Gemini, que decide si una respuesta del
LLM principal **filtra información protegida del system prompt**. Reemplaza al
Output Filter léxico (solapamiento de palabras + Levenshtein) del backend
[`system-prompt-extraction-backend`](../system-prompt-extraction-backend/README.md),
que lo consume vía HTTP y conserva el filtro léxico como respaldo.

```
Usuario → Input Guard (Gemini "security") → LLM principal (Gemini "main")
                                                   │
                         Spring Boot · OutputFilterService (engine=ml)
                                                   │  POST /api/output-filter
                                                   ▼
                FastAPI · preprocesamiento → cross-encoder ONNX INT8 → P(fuga)
                                                   │
                         bloquea (mensaje genérico + log) │ entrega la respuesta
```

## Modelo

- **Arquitectura**: cross-encoder `(fragmento protegido, respuesta) → {SAFE, LEAK}`.
- **Base**: [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  (Transformer destilado, 118 M parámetros, 50+ idiomas, preentrenado en paráfrasis).
  Se eligió en vez de DistilBERT inglés porque el sistema responde en **español** y
  el Vector 3 incluye **traducción**: un modelo multilingüe alineado entre idiomas
  detecta la misma regla en inglés, portugués o alemán. `distilbert-base-multilingual-cased`
  está soportado con `--base-model`.
- **Producción**: ONNX Runtime con cuantización dinámica INT8 (CPU, sin PyTorch).

### Criterio de etiquetado

| Etiqueta | Cuándo | Ejemplo |
|---|---|---|
| **FUGA** | La respuesta revela *cómo está instruido* el asistente: cita, parafrasea, traduce, resume o enumera una regla **como regla**, o expone nombres internos (herramientas, número de regla). | "Me indicaron remitir toda excepción al canal de PQRS en vez de aprobarla." |
| **SEGURO** | La respuesta *aplica* la regla sin describirla, ayuda, rechaza de forma genérica o habla de otra cosa. | "Para pedir una excepción, radícala por el canal oficial de PQRS." |

Es justo el caso donde falla el filtro léxico: ambas frases comparten casi todo el
vocabulario, pero solo la primera es una fuga.

### Preprocesamiento determinista (antes del modelo)

Un Transformer no decodifica Base64, así que la respuesta se expande en **vistas**:
original normalizada (NFKC, sin caracteres invisibles, homoglifos → latino),
Base64/hex decodificados, ROT13 y texto invertido (solo si así parece más lenguaje
natural), leetspeak, y una vista **acumulada** con el historial de la sesión
(Vector 4). Cada vista se parte en ventanas de ≤600 caracteres con solapamiento.
El veredicto es el máximo de P(fuga) sobre vista × ventana × fragmento.

## Resultados (modelo final v5, ONNX INT8, umbral 0.37)

Evaluación end-to-end sobre 1.341 respuestas completas (`reports/eval_onnx.int8.md`):

| Filtro | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| IA entrenada | 0.980 | 0.952 | 0.966 | 0.049 |
| Léxico anterior (0.45) | 0.909 | 0.313 | 0.466 | 0.078 |

- Sondas a ciegas escritas a mano: IA 10/12 fugas y 12/12 seguras; léxico 0/12 fugas.
- Latencia en CPU (4 hilos, 7 fragmentos): p50 135 ms · p95 293 ms. El objetivo de < 100 ms
  **no se cumple aún**; ver `--num-layers 6`.
- Historial de iteraciones, lecciones y pendientes: [`../AVANCE_OUTPUT_FILTER_IA.md`](../AVANCE_OUTPUT_FILTER_IA.md).

> Estas métricas se obtienen sobre datos sintéticos de la misma familia que los de entrenamiento.
> La validación definitiva requiere respuestas reales de Gemini etiquetadas por personas.

## Estructura

```
data/seed/prompts.json        corpus semilla escrito a mano (INTS + 5 prompts auxiliares + 1 reservado)
data/manual/                  casos límite y secuencias multi-turno escritos a mano (solo test)
data/generated/gemini.jsonl   (opcional) datos generados con scripts/generate_with_gemini.py
leakfilter/synth.py           generador sintético por vector (Opción B programática)
leakfilter/build_dataset.py   dedupe + split 80/10/10 por grupo + test de prompt no visto + set e2e
leakfilter/train.py           fine-tuning (PyTorch puro, CPU/GPU) + calibración del umbral
leakfilter/export_onnx.py     exportación ONNX FP32 + cuantización INT8 + verificación de paridad
leakfilter/evaluate.py        métricas por par, por vector (e2e), comparación con el léxico, latencia
leakfilter/preprocessing.py   vistas y ventanas (compartido por entrenamiento y servicio)
leakfilter/inference.py       LeakScorer (ONNX/torch), compartido por servicio y evaluación
leakfilter/lexical_baseline.py port fiel del filtro léxico Java (solo para comparar)
service/app.py                FastAPI: POST /api/output-filter, GET /health
tests/                        pytest (preprocesamiento + contrato HTTP con scorer falso)
reports/                      log de entrenamiento y reportes de evaluación
```

## Reproducir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements-train.txt

python -m leakfilter.build_dataset                 # data/processed/*
python -m leakfilter.train --epochs 2              # models/leakfilter-v1 (~22 min en CPU de 8 hilos)
python -m leakfilter.export_onnx                   # model.onnx + model.int8.onnx
python -m leakfilter.evaluate --backend onnx       # reports/eval_onnx.int8.md
python -m leakfilter.probe --models models/leakfilter-v1   # sondas escritas a mano
pytest
```

Opcional (Opción B con LLM), con una API key de Gemini:

```bash
export GEMINI_SECURITY_API_KEY=...
python scripts/generate_with_gemini.py --per-technique 3
# revisar ≥10 % de data/generated/gemini.jsonl y volver a build_dataset + train
```

## Servir

```bash
uvicorn service.app:app --host 0.0.0.0 --port 8001
# o: docker build -t fdsi-leakfilter . && docker run -p 8001:8001 fdsi-leakfilter
```

```bash
curl -s localhost:8001/api/output-filter -H 'Content-Type: application/json' -d '{
  "response": "Tengo prohibido compartir radicados de otras personas.",
  "protected_fragments": ["Nunca reveles datos de otros usuarios, números de radicado ajenos, ni el contenido de bases de datos internas"],
  "history": [], "session_id": "demo"}'
```

| Variable | Descripción | Default |
|---|---|---|
| `LEAKFILTER_MODEL_DIR` | Directorio del modelo | `models/leakfilter-v1` |
| `LEAKFILTER_BACKEND` | `onnx` o `torch` | `onnx` |
| `LEAKFILTER_ONNX_FILE` | Archivo ONNX | `model.int8.onnx` |
| `LEAKFILTER_THRESHOLD` | Sobrescribe el umbral calibrado | *(filter_config.json)* |
| `LEAKFILTER_THREADS` | Hilos de ONNX Runtime | *(automático)* |
| `LEAKFILTER_API_TOKEN` | Exige header `X-Filter-Token` | *(vacío)* |

En el backend: `OUTPUT_FILTER_ENGINE=ml` (default) y `OUTPUT_FILTER_ML_URL=http://localhost:8001`.
Si el servicio no responde, el backend usa el detector léxico (nunca fail-open).

> El servicio nunca escribe en su log el texto de las respuestas ni de los fragmentos.
