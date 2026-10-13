"""The whole pipeline in one object: corpus -> A -> SVD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .corpora import Corpus, load_corpus
from .model import LSA
from .text import TermDocumentMatrix, build_matrix


@dataclass
class Experiment:
    corpus: Corpus
    tdm: TermDocumentMatrix
    lsa: LSA

    @property
    def labels(self) -> np.ndarray | None:
        return np.array(self.corpus.labels) if self.corpus.labels else None

    def search(self, query: str, k: int) -> dict:
        """Rank documents for a query in raw TF-IDF space and in concept space."""
        q = self.tdm.query_vector(query)
        qc = self.tdm.query_counts(query)
        shared = [
            sorted(str(t) for t in self.tdm.terms[(qc > 0) & (self.tdm.counts[:, j] > 0)])
            for j in range(self.tdm.n_docs)
        ]
        return {
            "q": q,
            "known_terms": [str(t) for t in self.tdm.terms[qc > 0]],
            "q_hat": self.lsa.fold_in(q, k),
            "q_coords": self.lsa.query_coords(q, k),
            "raw": self.lsa.raw_similarity(q),
            "lsa": self.lsa.latent_similarity(q, k),
            "shared": shared,
        }


    def showcase_pair(self, k: int) -> tuple[int, int, float, float] | None:
        """Two documents with the same label that share no indexed words, yet sit
        closest together in concept space. Returns (i, j, raw_cos, lsa_cos)."""
        R = self.lsa.doc_similarity()
        L = self.lsa.doc_similarity(k)
        if self.corpus.showcase:
            titles = list(self.corpus.titles)
            i, j = (titles.index(t) for t in self.corpus.showcase)
            return i, j, float(R[i, j]), float(L[i, j])
        y = self.labels
        n = len(R)
        best = None
        overlap = (self.tdm.counts > 0).astype(int)
        shared = overlap.T @ overlap
        for i in range(n):
            for j in range(i + 1, n):
                if shared[i, j] or shared[i, i] == 0 or shared[j, j] == 0:
                    continue
                if y is not None and y[i] != y[j]:
                    continue
                # prefer documents with a few more words so the example is readable
                score = L[i, j] + 0.002 * min(shared[i, i], shared[j, j])
                if best is None or score > best[0]:
                    best = (score, i, j)
        if best is None:
            return None
        _, i, j = best
        return i, j, float(R[i, j]), float(L[i, j])


def run_experiment(corpus: Corpus | str, weighting: str | None = None) -> Experiment:
    if isinstance(corpus, str):
        corpus = load_corpus(corpus)
    tdm = build_matrix(corpus, weighting)
    return Experiment(corpus=corpus, tdm=tdm, lsa=LSA(tdm.A, tdm.terms))
