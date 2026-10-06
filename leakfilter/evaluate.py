"""Evaluación del filtro: nivel par, nivel respuesta (e2e) por vector, línea base léxica y latencia.

Uso:
  python -m leakfilter.evaluate --model-dir models/leakfilter-v1 --backend onnx
Genera reports/eval_<backend>.json y reports/eval_<backend>.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from leakfilter.inference import LeakScorer
from leakfilter.lexical_baseline import lexical_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
LEXICAL_THRESHOLD = 0.45

VECTOR_NAMES = {
    "1": "V1 Override directo → copia literal",
    "2": "V2 Role-play / autoridad ficticia",
    "3": "V3 Traducción / codificación",
    "4": "V4 Paráfrasis / fuga parcial / incremental",
    "5": "V5 Input Guard caído + traducción + Base64",
    "6": "V6 Homoglifos / caracteres invisibles",
    "7": "V7 Mezcla de idiomas",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def binary_metrics(labels, scores, threshold) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    out = {
        "n": int(len(labels)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "fpr": float(((preds == 1) & (labels == 0)).sum() / max(1, (labels == 0).sum())),
    }
    if len(set(labels.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(labels, scores))
        out["pr_auc"] = float(average_precision_score(labels, scores))
    return out


def pair_level(scorer: LeakScorer, name: str) -> dict:
    rows = read_jsonl(DATA / f"{name}.jsonl")
    scores = scorer.score_pairs([r["fragment"] for r in rows], [r["response"] for r in rows])
    result = {"overall": binary_metrics([r["label"] for r in rows], scores, scorer.threshold), "by_vector": {}}
    by_vector = defaultdict(list)
    for row, score in zip(rows, scores):
        if row["label"] == 1:
            by_vector[str(row["vector"])].append(score >= scorer.threshold)
    for vector, hits in sorted(by_vector.items()):
        result["by_vector"][vector] = {"n": len(hits), "tpr": float(np.mean(hits))}
    return result


def load_fragments() -> dict[str, list[str]]:
    corpus = json.loads((ROOT / "data" / "seed" / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    return {p["prompt_id"]: [f["text"] for f in p["fragments"]] for p in corpus}


def e2e(scorer: LeakScorer) -> dict:
    rows = read_jsonl(DATA / "e2e_test.jsonl")
    fragments = load_fragments()
    # Calentamiento: la primera inferencia de ONNX Runtime incluye inicialización.
    for row in rows[:5]:
        scorer.evaluate(row["response"], fragments[row["prompt_id"]], row["history"])

    ml_scores, lex_scores, latencies, records = [], [], [], []
    for row in rows:
        verdict = scorer.evaluate(row["response"], fragments[row["prompt_id"]], row["history"])
        lex = lexical_score(row["response"], fragments[row["prompt_id"]], row["history"])
        ml_scores.append(verdict.score)
        lex_scores.append(lex)
        latencies.append(verdict.latency_ms)
        records.append({**row, "ml_score": round(verdict.score, 4), "ml_view": verdict.matched_view,
                        "lexical_score": round(lex, 4), "pairs": verdict.pairs_evaluated})

    labels = [r["label"] for r in rows]
    summary = {
        "overall": {
            "ml": binary_metrics(labels, ml_scores, scorer.threshold),
            "lexical": binary_metrics(labels, lex_scores, LEXICAL_THRESHOLD),
        },
        "by_vector": {},
        "by_attack_type": {},
        "safe_by_type": {},
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "mean": float(statistics.mean(latencies)),
            "max": float(max(latencies)),
            "mean_pairs_per_response": float(statistics.mean(r["pairs"] for r in records)),
        },
        "latency_by_kind_ms": {},
        "threshold_sweep": [],
    }

    # Latencia separando respuestas en lenguaje natural (lo habitual) de las codificadas (ataques).
    for kind, keep in (("natural", lambda r: r["source"] != "synthetic_encoded"),
                       ("encoded", lambda r: r["source"] == "synthetic_encoded")):
        lat = [lt for lt, r in zip(latencies, records) if keep(r)]
        if lat:
            summary["latency_by_kind_ms"][kind] = {"n": len(lat), "p50": float(np.percentile(lat, 50)),
                                                   "p95": float(np.percentile(lat, 95))}

    for threshold in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        m = binary_metrics(labels, ml_scores, threshold)
        summary["threshold_sweep"].append({"threshold": threshold, **{k: m[k] for k in ("precision", "recall", "f1", "fpr")}})

    def bucket(key_fn, positives: bool):
        groups = defaultdict(lambda: {"ml": [], "lex": []})
        for rec in records:
            if (rec["label"] == 1) != positives:
                continue
            g = groups[key_fn(rec)]
            g["ml"].append(rec["ml_score"] >= scorer.threshold)
            g["lex"].append(rec["lexical_score"] >= LEXICAL_THRESHOLD)
        return {k: {"n": len(v["ml"]), "ml": float(np.mean(v["ml"])), "lexical": float(np.mean(v["lex"]))}
                for k, v in sorted(groups.items())}

    summary["by_vector"] = bucket(lambda r: str(r["vector"]), positives=True)  # TPR por vector
    summary["by_attack_type"] = bucket(lambda r: r["attack_type"], positives=True)
    summary["safe_by_type"] = bucket(lambda r: r["attack_type"], positives=False)  # FPR por tipo
    summary["errors"] = {
        "false_negatives": [r for r in records if r["label"] == 1 and r["ml_score"] < scorer.threshold],
        "false_positives": [r for r in records if r["label"] == 0 and r["ml_score"] >= scorer.threshold],
    }
    summary["records"] = records
    return summary


def to_markdown(report: dict) -> str:
    t = report["threshold"]
    lines = [f"# Evaluación — backend `{report['backend']}` · umbral {t}", ""]
    lines.append(f"Ventana: {report['window_chars']} caracteres")
    for name in report["pair_level"]:
        m = report["pair_level"][name]["overall"]
        lines.append(f"**Nivel par · {name}** (n={m['n']}): precision {m['precision']:.3f} · recall {m['recall']:.3f}"
                     f" · F1 {m['f1']:.3f} · FPR {m['fpr']:.3f} · ROC-AUC {m.get('roc_auc', 0):.3f}")
    e = report["e2e"]
    lines += ["", "## End-to-end (respuesta completa, pipeline real)", "",
              "| Filtro | Precision | Recall | F1 | FPR |", "|---|---|---|---|---|"]
    for k in ("ml", "lexical"):
        m = e["overall"][k]
        lines.append(f"| {'IA entrenada' if k == 'ml' else 'Léxico actual (0.45)'} | {m['precision']:.3f} | "
                     f"{m['recall']:.3f} | {m['f1']:.3f} | {m['fpr']:.3f} |")
    lines += ["", "### TPR por vector", "", "| Vector | n | IA | Léxico |", "|---|---|---|---|"]
    for v, m in e["by_vector"].items():
        lines.append(f"| {VECTOR_NAMES.get(v, v)} | {m['n']} | {m['ml']:.1%} | {m['lexical']:.1%} |")
    lines += ["", "### TPR por tipo de ataque", "", "| Tipo | n | IA | Léxico |", "|---|---|---|---|"]
    for k, m in e["by_attack_type"].items():
        lines.append(f"| {k} | {m['n']} | {m['ml']:.1%} | {m['lexical']:.1%} |")
    lines += ["", "### Tasa de bloqueo en respuestas SEGURAS (FPR por tipo)", "", "| Tipo | n | IA | Léxico |",
              "|---|---|---|---|"]
    for k, m in e["safe_by_type"].items():
        lines.append(f"| {k} | {m['n']} | {m['ml']:.1%} | {m['lexical']:.1%} |")
    lat = e["latency_ms"]
    lines += ["", f"**Latencia por respuesta** (CPU, {report['threads']} hilos): p50 {lat['p50']:.1f} ms · "
              f"p95 {lat['p95']:.1f} ms · p99 {lat['p99']:.1f} ms · pares/respuesta {lat['mean_pairs_per_response']:.1f}"]
    for kind, m in e["latency_by_kind_ms"].items():
        lines.append(f"- {kind} (n={m['n']}): p50 {m['p50']:.1f} ms · p95 {m['p95']:.1f} ms")
    lines += ["", "### Barrido de umbral (e2e)", "", "| Umbral | Precision | Recall | F1 | FPR |", "|---|---|---|---|---|"]
    for row in e["threshold_sweep"]:
        lines.append(f"| {row['threshold']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['fpr']:.3f} |")
    lines += ["", "## Errores del clasificador", ""]
    for kind in ("false_negatives", "false_positives"):
        lines.append(f"**{kind}** ({len(e['errors'][kind])}):")
        for r in e["errors"][kind][:25]:
            lines.append(f"- [{r['attack_type']}] score={r['ml_score']} — {r['response'][:140]!r}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=str(ROOT / "models" / "leakfilter-v1"))
    parser.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    parser.add_argument("--onnx-file", default="model.int8.onnx")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--window-chars", type=int, default=None)
    parser.add_argument("--e2e-only", action="store_true", help="omite la evaluación por pares")
    parser.add_argument("--tag", default="", help="sufijo para los archivos del reporte")
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    scorer = LeakScorer(args.model_dir, backend=args.backend, onnx_file=args.onnx_file,
                        threshold=args.threshold, intra_op_threads=args.threads, window_chars=args.window_chars)
    report = {
        "backend": args.backend if args.backend == "torch" else f"onnx:{args.onnx_file}",
        "threshold": scorer.threshold,
        "threads": args.threads,
        "window_chars": scorer.window_chars,
        "pair_level": {} if args.e2e_only else {name: pair_level(scorer, name) for name in ("test", "test_unseen_prompt")},
        "e2e": e2e(scorer),
    }
    suffix = "torch" if args.backend == "torch" else Path(args.onnx_file).stem.replace("model", "onnx")
    suffix += args.tag
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    records = report["e2e"].pop("records")
    with (out / f"e2e_records_{suffix}.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / f"eval_{suffix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"eval_{suffix}.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))


if __name__ == "__main__":
    main()
