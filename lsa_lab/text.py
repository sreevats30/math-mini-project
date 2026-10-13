"""Step 1: turn documents into the term-document matrix A (m terms x n documents).

The weighting is written out explicitly (instead of calling TfidfVectorizer) so
every number shown in the app and on the slides can be traced to a formula:

    tf(t, d)  = 1 + ln(count(t, d))   if count > 0, else 0     (sublinear term frequency)
    idf(t)    = ln((1 + n) / (1 + df(t))) + 1                   (smoothed inverse document frequency)
    w(t, d)   = tf(t, d) * idf(t)
    A[:, d]   = w(:, d) / ||w(:, d)||₂                           (each document has unit length)

tests/test_lsa.py checks that this matches scikit-learn's
TfidfVectorizer(sublinear_tf=True) exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .corpora import Corpus

# Words of two or more letters; numbers and punctuation are dropped.
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"

WEIGHTINGS = {
    "tfidf": "TF-IDF (log tf × idf, unit-length documents)",
    "count": "Raw word counts",
}


@dataclass
class TermDocumentMatrix:
    corpus: Corpus
    weighting: str
    vectorizer: CountVectorizer
    terms: np.ndarray        # (m,)  vocabulary in column order of the vectorizer
    counts: np.ndarray       # (m, n) raw counts, terms x documents
    df: np.ndarray           # (m,)  number of documents containing each term
    idf: np.ndarray          # (m,)  smoothed idf (ones for raw counts)
    A: np.ndarray            # (m, n) weighted matrix used for the SVD

    @property
    def shape(self) -> tuple[int, int]:
        return self.A.shape

    @property
    def n_terms(self) -> int:
        return self.A.shape[0]

    @property
    def n_docs(self) -> int:
        return self.A.shape[1]

    @property
    def sparsity(self) -> float:
        """Fraction of entries in A that are exactly zero."""
        return float(np.mean(self.A == 0))

    def term_index(self, term: str) -> int | None:
        hits = np.flatnonzero(self.terms == term.lower())
        return int(hits[0]) if hits.size else None

    def tokens(self, text: str) -> list[str]:
        """Tokens the vectorizer keeps for this text (lower-cased, stop words removed)."""
        tokens = self.vectorizer.build_analyzer()(text)
        vocab = self.vectorizer.vocabulary_
        return [t for t in tokens if t in vocab]

    def tf(self, counts: np.ndarray) -> np.ndarray:
        if self.weighting == "count":
            return counts.astype(float)
        out = np.zeros_like(counts, dtype=float)
        nz = counts > 0
        out[nz] = 1.0 + np.log(counts[nz])
        return out

    def weight(self, counts: np.ndarray) -> np.ndarray:
        """Apply this matrix's weighting to a count vector or matrix (terms in rows)."""
        counts = np.asarray(counts, dtype=float)
        if self.weighting == "count":
            return counts.copy()
        w = self.tf(counts) * (self.idf[:, None] if counts.ndim == 2 else self.idf)
        return _unit_columns(w)

    def query_counts(self, text: str) -> np.ndarray:
        return np.asarray(self.vectorizer.transform([text]).toarray()).ravel().astype(float)

    def query_vector(self, text: str) -> np.ndarray:
        """A new piece of text expressed in the same term space as the columns of A."""
        return self.weight(self.query_counts(text))

    def tfidf_breakdown(self, term: str, doc: int) -> dict:
        """Every intermediate number behind one entry A[term, doc]."""
        i = self.term_index(term)
        if i is None:
            raise KeyError(term)
        c = float(self.counts[i, doc])
        n = self.n_docs
        tf = float(self.tf(np.array([c]))[0])
        w_col = self.tf(self.counts[:, doc]) * self.idf
        norm = float(np.linalg.norm(w_col))
        return {
            "count": c,
            "tf": tf,
            "df": int(self.df[i]),
            "n": n,
            "idf": float(self.idf[i]),
            "w": tf * float(self.idf[i]),
            "norm": norm,
            "a": float(self.A[i, doc]),
        }


def _unit_columns(W: np.ndarray) -> np.ndarray:
    if W.ndim == 1:
        nrm = np.linalg.norm(W)
        return W / nrm if nrm > 0 else W
    nrm = np.linalg.norm(W, axis=0)
    nrm[nrm == 0] = 1.0
    return W / nrm


def build_matrix(corpus: Corpus, weighting: str | None = None) -> TermDocumentMatrix:
    weighting = weighting or corpus.weighting
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {list(WEIGHTINGS)}")

    min_df = corpus.min_df
    # A min_df above 1 makes no sense for tiny corpora.
    if corpus.n_docs < 20:
        min_df = 1
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        stop_words=corpus.stop_words,
        min_df=min_df,
        max_df=corpus.max_df,
        vocabulary=list(corpus.vocabulary) if corpus.vocabulary else None,
    )
    X = vectorizer.fit_transform(corpus.texts)            # n x m sparse counts
    counts = np.asarray(X.toarray(), dtype=float).T       # m x n
    terms = np.array(vectorizer.get_feature_names_out())

    n = counts.shape[1]
    df = (counts > 0).sum(axis=1).astype(int)
    if weighting == "count":
        idf = np.ones(len(terms))
    else:
        idf = np.log((1.0 + n) / (1.0 + df)) + 1.0

    tdm = TermDocumentMatrix(
        corpus=corpus,
        weighting=weighting,
        vectorizer=vectorizer,
        terms=terms,
        counts=counts,
        df=df,
        idf=idf,
        A=np.zeros_like(counts),
    )
    tdm.A = tdm.weight(counts)
    return tdm
