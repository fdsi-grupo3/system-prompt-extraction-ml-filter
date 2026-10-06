"""Fine-tuning del cross-encoder (fragmento protegido, respuesta) → {seguro, fuga}.

Bucle de entrenamiento en PyTorch puro (sin Trainer) para no depender de cambios de
API entre versiones de transformers y para que corra igual en CPU o GPU.

Uso:
  python -m leakfilter.train                              # MiniLM multilingüe (default)
  python -m leakfilter.train --base-model distilbert-base-multilingual-cased --out models/distilbert-v1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score, fbeta_score, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
DEFAULT_BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_loader(rows, tokenizer, max_length, batch_size, shuffle):
    def collate(batch):
        enc = tokenizer([r["fragment"] for r in batch], [r["response"] for r in batch],
                        truncation="only_second", max_length=max_length, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([r["label"] for r in batch], dtype=torch.long)
        return enc

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.inference_mode()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for batch in loader:
        labels.append(batch.pop("labels").numpy())
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
    }


def calibrate_threshold(probs: np.ndarray, labels: np.ndarray, min_recall: float) -> tuple[float, str]:
    """Umbral que maximiza F2 en validación (recall pesa el doble que precision).

    En este filtro un falso negativo (fuga entregada) es peor que un falso positivo
    (respuesta legítima reemplazada por el mensaje genérico), así que se prioriza recall.
    Como el modelo produce probabilidades muy polarizadas, suele haber una MESETA de umbrales
    con el mismo F2; se toma su punto medio, que es más robusto que cualquiera de los extremos.
    min_recall solo se usa para advertir si ni el mejor umbral lo alcanza.
    """
    grid = [round(float(t), 2) for t in np.linspace(0.05, 0.95, 91)]
    f2 = {t: fbeta_score(labels, (probs >= t).astype(int), beta=2, zero_division=0) for t in grid}
    best_f2 = max(f2.values())
    plateau = [t for t in grid if f2[t] >= best_f2 - 1e-9]
    threshold = plateau[len(plateau) // 2]
    recall = recall_score(labels, (probs >= threshold).astype(int), zero_division=0)
    policy = (f"punto medio de la meseta de máximo F2 en validación ({plateau[0]}-{plateau[-1]}), nivel par")
    if recall < min_recall:
        policy += f"; ADVERTENCIA: recall {recall:.3f} < {min_recall}"
    return threshold, policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--out", default=str(ROOT / "models" / "leakfilter-v1"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--min-recall", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-layers", type=int, default=None,
                        help="poda del encoder: conserva N capas equiespaciadas (menor latencia)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="no entrena: recalcula el umbral del modelo ya guardado en --out")
    args = parser.parse_args()

    if args.recalibrate:
        recalibrate(Path(args.out), args.min_recall)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} threads={torch.get_num_threads()} base={args.base_model}")

    train_rows, val_rows = read_jsonl(DATA / "train.jsonl"), read_jsonl(DATA / "val.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label={0: "SAFE", 1: "LEAK"}, label2id={"SAFE": 0, "LEAK": 1}
    )
    if args.num_layers:
        prune_layers(model, args.num_layers)
    model = model.to(device)

    train_loader = make_loader(train_rows, tokenizer, args.max_length, args.batch_size, shuffle=True)
    val_loader = make_loader(val_rows, tokenizer, args.max_length, 64, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * args.warmup), total_steps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_score = [], -1.0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 25 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={running / step:.4f} "
                      f"elapsed={time.time() - started:.0f}s", flush=True)

        probs, labels = predict(model, val_loader, device)
        val = metrics(probs, labels, 0.5)
        val.update(epoch=epoch, train_loss=running / len(train_loader))
        history.append(val)
        print(f"epoch {epoch}: {json.dumps(val)}", flush=True)

        if val["pr_auc"] > best_score:  # PR-AUC no depende del umbral
            best_score = val["pr_auc"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            best_probs, best_labels = probs, labels

    threshold, policy = calibrate_threshold(best_probs, best_labels, args.min_recall)
    config = {
        "base_model": args.base_model,
        "num_layers": model.config.num_hidden_layers,
        "max_length": args.max_length,
        "threshold": threshold,
        "threshold_policy": policy,
        "val_at_threshold": metrics(best_probs, best_labels, threshold),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "training_seconds": round(time.time() - started, 1),
        "device": str(device),
        "history": history,
        "label_policy": "v1: revelar cómo está instruido el asistente = fuga; aplicar la regla sin describirla = seguro",
    }
    (out_dir / "filter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"guardado en {out_dir} · umbral calibrado={threshold} · {config['training_seconds']}s")


def prune_layers(model, num_layers: int) -> None:
    """Conserva num_layers capas equiespaciadas del encoder (p. ej. 6 de 12: 0, 2, 4, ...).

    Cada capa cuesta lo mismo en inferencia, así que la latencia baja casi en proporción;
    el fine-tuning posterior recupera la mayor parte de la calidad.
    """
    encoder = model.base_model.encoder
    total = len(encoder.layer)
    keep = sorted({round(i * (total - 1) / (num_layers - 1)) for i in range(num_layers)}) if num_layers > 1 else [0]
    encoder.layer = torch.nn.ModuleList(encoder.layer[i] for i in keep)
    model.config.num_hidden_layers = len(keep)
    print(f"encoder podado: capas {keep} de {total}")


def recalibrate(out_dir: Path, min_recall: float) -> None:
    config_path = out_dir / "filter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    loader = make_loader(read_jsonl(DATA / "val.jsonl"), tokenizer, config["max_length"], 64, shuffle=False)
    probs, labels = predict(model, loader, torch.device("cpu"))
    threshold, policy = calibrate_threshold(probs, labels, min_recall)
    config.update(threshold=threshold, threshold_policy=policy, val_at_threshold=metrics(probs, labels, threshold))
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"umbral recalibrado={threshold} ({policy}) · {config['val_at_threshold']}")


if __name__ == "__main__":
    main()
