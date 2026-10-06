"""Inferencia del clasificador de fugas, compartida por el servicio FastAPI y la evaluación.

Flujo para una respuesta del LLM:
  1. build_views(): original normalizada + vistas decodificadas (Base64, hex, ROT13,
     invertido, leet) + vista acumulada con el historial de la sesión (Vector 4).
  2. split_windows(): cada vista se parte en ventanas que caben en el modelo.
  3. Se arma un lote con todos los pares (fragmento protegido, ventana) y se calcula
     P(fuga) con el cross-encoder.
  4. El veredicto es el máximo de P(fuga); is_leak = máximo >= umbral.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from leakfilter.preprocessing import DEFAULT_WINDOW_CHARS, build_views, split_windows


@dataclass
class LeakVerdict:
    is_leak: bool
    score: float
    threshold: float
    matched_fragment: str | None
    matched_view: str | None
    accumulated: bool
    pairs_evaluated: int
    latency_ms: float


def _softmax_leak(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp[:, 1] / exp.sum(axis=1)


class LeakScorer:
    """Carga el modelo (ONNX o PyTorch) y evalúa respuestas contra fragmentos protegidos."""

    def __init__(self, model_dir: str | Path, backend: str = "onnx", onnx_file: str = "model.int8.onnx",
                 threshold: float | None = None, max_batch: int = 32, intra_op_threads: int | None = None,
                 window_chars: int | None = None):
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        config_path = self.model_dir / "filter_config.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        self.threshold = threshold if threshold is not None else float(self.config.get("threshold", 0.5))
        self.max_length = int(self.config.get("max_length", 256))
        self.window_chars = window_chars or int(self.config.get("window_chars", DEFAULT_WINDOW_CHARS))
        self.max_batch = max_batch
        self.backend = backend
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        if backend == "onnx":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if intra_op_threads:
                options.intra_op_num_threads = intra_op_threads
            self.session = ort.InferenceSession(str(self.model_dir / onnx_file), options,
                                                providers=["CPUExecutionProvider"])
            self.input_names = {i.name for i in self.session.get_inputs()}
        elif backend == "torch":
            import torch
            from transformers import AutoModelForSequenceClassification

            self.torch = torch
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).eval()
        else:
            raise ValueError(f"backend desconocido: {backend}")

    # ------------------------------------------------------------------ pares sueltos
    def score_pairs(self, fragments: list[str], responses: list[str]) -> np.ndarray:
        """P(fuga) para cada par (fragments[i], responses[i])."""
        scores = []
        for start in range(0, len(fragments), self.max_batch):
            frag_batch = fragments[start:start + self.max_batch]
            resp_batch = responses[start:start + self.max_batch]
            if self.backend == "onnx":
                enc = self.tokenizer(frag_batch, resp_batch, truncation="only_second",
                                     max_length=self.max_length, padding=True, return_tensors="np")
                feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
                logits = self.session.run(None, feeds)[0]
            else:
                enc = self.tokenizer(frag_batch, resp_batch, truncation="only_second",
                                     max_length=self.max_length, padding=True, return_tensors="pt")
                with self.torch.inference_mode():
                    logits = self.model(**enc).logits.numpy()
            scores.append(_softmax_leak(logits))
        return np.concatenate(scores) if scores else np.zeros(0)

    # ------------------------------------------------------------------ respuesta completa
    def evaluate(self, response: str, fragments: list[str], history: list[str] | None = None) -> LeakVerdict:
        start = time.perf_counter()
        pair_frags, pair_texts, pair_meta = [], [], []
        for view in build_views(response, history):
            for window in split_windows(view.text, self.window_chars):
                for fragment in fragments:
                    pair_frags.append(fragment)
                    pair_texts.append(window)
                    pair_meta.append((fragment, view.name))

        scores = self.score_pairs(pair_frags, pair_texts)
        best = int(np.argmax(scores)) if len(scores) else -1
        score = float(scores[best]) if best >= 0 else 0.0
        fragment, view_name = pair_meta[best] if best >= 0 else (None, None)
        is_leak = score >= self.threshold
        return LeakVerdict(
            is_leak=is_leak,
            score=score,
            threshold=self.threshold,
            matched_fragment=fragment if is_leak else None,
            matched_view=view_name,
            accumulated=is_leak and view_name == "accumulated",
            pairs_evaluated=len(scores),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
