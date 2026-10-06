"""Exporta el modelo entrenado a ONNX (FP32) y lo cuantiza dinámicamente a INT8.

La cuantización INT8 dinámica reduce el tamaño ~4x y acelera la inferencia en CPU
sin necesitar datos de calibración. Se verifica que las probabilidades de ambos
modelos coincidan con PyTorch en un lote de ejemplos de validación.

Uso:  python -m leakfilter.export_onnx --model-dir models/leakfilter-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


class _LogitsOnly(torch.nn.Module):
    """Expone solo los logits, con las entradas que produce el tokenizer (con o sin token_type_ids)."""

    def __init__(self, model, input_names):
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, *inputs):
        return self.model(**dict(zip(self.input_names, inputs))).logits


def _softmax_leak(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp[:, 1] / exp.sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=str(ROOT / "models" / "leakfilter-v1"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    sample = tokenizer(["fragmento protegido de ejemplo"], ["respuesta de ejemplo del asistente"],
                       return_tensors="pt", padding=True)
    input_names = list(sample.keys())
    wrapper = _LogitsOnly(model, input_names).eval()
    fp32_path = model_dir / "model.onnx"
    int8_path = model_dir / "model.int8.onnx"

    torch.onnx.export(
        wrapper,
        tuple(sample[name] for name in input_names),
        str(fp32_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes={name: {0: "batch", 1: "sequence"} for name in input_names} | {"logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    # Verificación de paridad contra PyTorch.
    rows = [json.loads(line) for line in (ROOT / "data" / "processed" / "val.jsonl").read_text(encoding="utf-8").splitlines()[:64]]
    enc = tokenizer([r["fragment"] for r in rows], [r["response"] for r in rows], truncation="only_second",
                    max_length=256, padding=True, return_tensors="pt")
    with torch.inference_mode():
        reference = _softmax_leak(model(**enc).logits.numpy())

    report = {}
    for name, path in (("fp32", fp32_path), ("int8", int8_path)):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        logits = session.run(None, {k: v.numpy().astype(np.int64) for k, v in enc.items()})[0]
        probs = _softmax_leak(logits)
        report[name] = {
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "max_abs_diff_vs_torch": float(np.abs(probs - reference).max()),
            "decision_agreement@0.5": float(((probs >= 0.5) == (reference >= 0.5)).mean()),
        }
    (model_dir / "onnx_export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
