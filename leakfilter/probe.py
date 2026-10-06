"""Evalúa modelos sobre sondas escritas a mano, fuera de toda plantilla del generador.

El e2e sintético comparte estructura con el entrenamiento y tiende a ser optimista;
las sondas miden generalización a fraseos realmente nuevos.
  data/manual/probes.jsonl        sondas v1 (diagnóstico; se usaron para diseñar la iteración 4)
  data/manual/probes_blind.jsonl  sondas a ciegas (escritas ANTES de entrenar la iteración 4)

Uso:  python -m leakfilter.probe --models models/leakfilter-v1 models/leakfilter-v1-iter2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leakfilter.inference import LeakScorer
from leakfilter.lexical_baseline import lexical_score

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--files", nargs="+", default=["data/manual/probes.jsonl", "data/manual/probes_blind.jsonl"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    corpus = json.loads((ROOT / "data" / "seed" / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    fragments = [f["text"] for p in corpus if p["prompt_id"] == "ints" for f in p["fragments"]]
    scorers = {Path(m).name: LeakScorer(m, intra_op_threads=4) for m in args.models}

    summary = {}
    for file in args.files:
        probes = [json.loads(line) for line in (ROOT / file).read_text(encoding="utf-8").splitlines() if line.strip()]
        result = {name: {"leaks_detected": 0, "safe_passed": 0} for name in [*scorers, "lexical"]}
        n_leak = sum(p["label"] for p in probes)
        for probe in probes:
            decisions = {name: s.evaluate(probe["response"], fragments) for name, s in scorers.items()}
            lex = lexical_score(probe["response"], fragments)
            flags = {name: v.is_leak for name, v in decisions.items()} | {"lexical": lex >= 0.45}
            for name, flagged in flags.items():
                key = "leaks_detected" if probe["label"] == 1 else "safe_passed"
                result[name][key] += int(flagged == bool(probe["label"]))
            if args.verbose:
                scores = " ".join(f"{n}={v.score:.2f}" for n, v in decisions.items())
                print(f"{probe['id']} label={probe['label']} {scores} lexical={lex:.2f}")
        for name, r in result.items():
            r["text"] = f"fugas {r['leaks_detected']}/{n_leak} · seguras {r['safe_passed']}/{len(probes) - n_leak}"
        summary[file] = result
        print(f"\n{file}")
        for name, r in result.items():
            print(f"  {name:<24} {r['text']}")

    out = ROOT / "reports" / "probes.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
