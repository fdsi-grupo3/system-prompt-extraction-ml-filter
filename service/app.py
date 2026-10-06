"""Servicio FastAPI del Output Filter basado en IA.

Endpoints:
  POST /api/output-filter   evalúa una respuesta del LLM principal contra los fragmentos protegidos
  GET  /health              estado del modelo cargado

Variables de entorno:
  LEAKFILTER_MODEL_DIR       directorio del modelo exportado (default: models/leakfilter-v1)
  LEAKFILTER_BACKEND         onnx | torch (default: onnx)
  LEAKFILTER_ONNX_FILE       archivo ONNX dentro del directorio (default: model.int8.onnx)
  LEAKFILTER_THRESHOLD       sobrescribe el umbral calibrado en filter_config.json
  LEAKFILTER_THREADS         hilos de ONNX Runtime (default: los que decida ORT)
  LEAKFILTER_API_TOKEN       si se define, exige el header X-Filter-Token con ese valor
  LEAKFILTER_FRAGMENTS_FILE  JSON con fragmentos por defecto (si el request no envía ninguno)

Por privacidad, el servicio nunca escribe en el log el texto de la respuesta ni de
los fragmentos: solo session_id, score, vista y latencia.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from leakfilter.inference import LeakScorer

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("leakfilter.service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class FilterRequest(BaseModel):
    response: str = Field(..., min_length=1, max_length=20000, description="Respuesta candidata del LLM principal")
    protected_fragment: str | None = Field(None, max_length=2000)
    protected_fragments: list[str] | None = Field(None, max_length=50)
    history: list[str] = Field(default_factory=list, max_length=10,
                               description="Respuestas ya entregadas en la sesión (Vector 4, acumulado)")
    session_id: str | None = Field(None, max_length=100)


class FilterResponse(BaseModel):
    is_leak: bool
    score: float
    threshold: float
    matched_fragment: str | None
    matched_view: str | None
    accumulated: bool
    pairs_evaluated: int
    latency_ms: float
    model_version: str


def _default_fragments() -> list[str]:
    path = os.getenv("LEAKFILTER_FRAGMENTS_FILE")
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data["fragments"]
    corpus = json.loads((ROOT / "data" / "seed" / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    return [f["text"] for p in corpus if p["prompt_id"] == "ints" for f in p["fragments"]]


def _load_scorer() -> LeakScorer:
    threshold = os.getenv("LEAKFILTER_THRESHOLD")
    threads = os.getenv("LEAKFILTER_THREADS")
    return LeakScorer(
        os.getenv("LEAKFILTER_MODEL_DIR", str(ROOT / "models" / "leakfilter-v1")),
        backend=os.getenv("LEAKFILTER_BACKEND", "onnx"),
        onnx_file=os.getenv("LEAKFILTER_ONNX_FILE", "model.int8.onnx"),
        threshold=float(threshold) if threshold else None,
        intra_op_threads=int(threads) if threads else None,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "scorer"):  # las pruebas inyectan un scorer falso
        app.state.scorer = _load_scorer()
        app.state.default_fragments = _default_fragments()
        app.state.model_version = Path(app.state.scorer.model_dir).name
        app.state.scorer.evaluate("warm-up", app.state.default_fragments[:1])
    log.info("modelo listo version=%s umbral=%.2f", app.state.model_version, app.state.scorer.threshold)
    yield


app = FastAPI(title="FDSI-GP-03 · Output Filter IA", version="1.0.0", lifespan=lifespan)


def require_token(x_filter_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("LEAKFILTER_API_TOKEN")
    if expected and not hmac.compare_digest(x_filter_token or "", expected):
        raise HTTPException(status_code=401, detail="token inválido")


@app.get("/health")
def health(request: Request) -> dict:
    scorer = request.app.state.scorer
    return {"status": "UP", "model_version": request.app.state.model_version,
            "backend": scorer.backend, "threshold": scorer.threshold}


@app.post("/api/output-filter", response_model=FilterResponse, dependencies=[Depends(require_token)])
def output_filter(body: FilterRequest, request: Request) -> FilterResponse:
    fragments = list(body.protected_fragments or [])
    if body.protected_fragment:
        fragments.append(body.protected_fragment)
    if not fragments:
        fragments = request.app.state.default_fragments

    verdict = request.app.state.scorer.evaluate(body.response, fragments, body.history)
    log.info("session=%s is_leak=%s score=%.3f view=%s pairs=%d latency_ms=%.1f", body.session_id,
             verdict.is_leak, verdict.score, verdict.matched_view, verdict.pairs_evaluated, verdict.latency_ms)
    return FilterResponse(**verdict.__dict__, model_version=request.app.state.model_version)
