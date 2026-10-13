"""Run the full analysis once and record every number used in the slides and README."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import clustering as cl
from .pipeline import Experiment, run_experiment

ROOT = Path(__file__).resolve().parent.parent
SWEEP_KS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]


def _search_examples(ex: Experiment, k: int, top: int = 5) -> list[dict]:
    out = []
    for q in ex.corpus.example_queries:
        r = ex.search(q, k)
        order = np.argsort(-r["lsa"])[:top]
        out.append({
            "query": q,
            "known_terms": r["known_terms"],
            "raw_matches": int(np.sum(r["raw"] > 0)),
            "top": [
                {
                    "title": ex.corpus.titles[j],
                    "label": ex.corpus.labels[j] if ex.corpus.labels else None,
                    "lsa": round(float(r["lsa"][j]), 3),
                    "raw": round(float(r["raw"][j]), 3),
                    "shared_words": r["shared"][j],
                }
                for j in order
            ],
        })
    return out


def build(verbose: bool = True) -> dict:
    t0 = time.time()
    exps = {key: run_experiment(key) for key in ("classic", "science", "news")}
    results: dict = {"corpora": {}}
    comps = {}
    for key, ex in exps.items():
        k = ex.corpus.default_k
        lsa = ex.lsa
        comp = cl.compare(lsa, ex.labels, k, len(ex.corpus.categories))
        comps[key] = comp
        pair = ex.showcase_pair(k)
        results["corpora"][key] = {
            "name": ex.corpus.name,
            "documents": ex.tdm.n_docs,
            "terms": ex.tdm.n_terms,
            "weighting": ex.tdm.weighting,
            "sparsity": round(ex.tdm.sparsity, 4),
            "rank": lsa.rank,
            "k": k,
            "singular_values_top10": [round(float(x), 4) for x in lsa.s[:10]],
            "eckart_young": {a: (round(b, 6) if isinstance(b, float) else b) for a, b in lsa.eckart_young(k).items()},
            "svd_checks": lsa.checks(),
            "clustering": comp.summary(),
            "word_pairs": [
                {"pair": [a, b], "corr_A": lsa.term_correlation(a, b), "corr_Ak": lsa.term_correlation(a, b, k)}
                for a, b in ex.corpus.word_pairs
            ],
            "showcase_pair": None if pair is None else {
                "a": ex.corpus.titles[pair[0]], "b": ex.corpus.titles[pair[1]],
                "raw_cosine": pair[2], "lsa_cosine": pair[3],
            },
            "search": _search_examples(ex, k),
            "concepts": [
                {"sigma": c.sigma, "positive": [t for t, _ in c.positive[:6]], "negative": [t for t, _ in c.negative[:6]]}
                for c in (lsa.concept(i, 6) for i in range(min(k, 6)))
            ],
        }
        if verbose:
            s = comp.summary()
            print(f"[{key:8}] A {ex.tdm.n_terms}x{ex.tdm.n_docs}, k={k}, energy kept "
                  f"{lsa.eckart_young(k)['energy_kept']:.1%} | ARI raw {s['raw']['ARI']['mean']:.3f} "
                  f"-> LSA {s['lsa']['ARI']['mean']:.3f}")
    news = exps["news"]
    sweep = cl.k_sweep(news.lsa, news.labels, 4, SWEEP_KS)
    results["news_k_sweep"] = sweep
    results["elapsed_seconds"] = round(time.time() - t0, 1)
    return {"results": results, "experiments": exps, "comparisons": comps, "sweep": sweep}


def save(results: dict, path: Path | None = None) -> Path:
    path = path or ROOT / "results" / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(results, indent=2, default=default, ensure_ascii=False), encoding="utf-8")
    return path
