"""Steps 2-4: factorise A with the SVD, truncate to rank k, and work in concept space.

Notation (thin SVD, r = min(m, n)):

    A   (m x n)  term-document matrix, documents are columns
    U   (m x r)  left singular vectors  = term loadings of each concept, UᵀU = I
    Σ   (r x r)  singular values σ₁ ≥ σ₂ ≥ … ≥ 0 = strength of each concept
    Vᵀ  (r x n)  right singular vectors = document loadings of each concept, VᵀV = I

    A = U Σ Vᵀ = Σᵢ σᵢ uᵢ vᵢᵀ          (a sum of rank-1 "concept layers")
    A_k = U_k Σ_k V_kᵀ                  (keep the k strongest layers)

Coordinates used throughout:

    document j in concept space:  d̂ⱼ = U_kᵀ aⱼ = Σ_k vⱼ    -> rows of V_k Σ_k   (n x k)
    term i in concept space:      t̂ᵢ = Σ_k uᵢ               -> rows of U_k Σ_k   (m x k)
    new query q (folding-in):     q̂  = Σ_k⁻¹ U_kᵀ q          (a new row of V_k)
                                  compared with documents as Σ_k q̂ = U_kᵀ q

Both a document already in A and a new query are mapped by the same matrix U_kᵀ,
so their cosine similarity is a fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cosine_rows(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of X and the vector y (0 when a norm is 0)."""
    xn = np.linalg.norm(X, axis=1)
    yn = np.linalg.norm(y)
    out = np.zeros(X.shape[0])
    ok = (xn > 0) & (yn > 0)
    out[ok] = (X[ok] @ y) / (xn[ok] * yn)
    return out


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between the rows of X."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Xn = X / n
    return np.clip(Xn @ Xn.T, -1.0, 1.0)


@dataclass
class Concept:
    index: int                      # 0-based
    sigma: float
    positive: list[tuple[str, float]]
    negative: list[tuple[str, float]]
    top_docs: list[tuple[int, float]]
    bottom_docs: list[tuple[int, float]]


class LSA:
    def __init__(self, A: np.ndarray, terms: np.ndarray):
        self.A = np.asarray(A, dtype=float)
        self.terms = np.asarray(terms)
        U, s, Vt = np.linalg.svd(self.A, full_matrices=False)
        U, Vt = self._fix_signs(U, Vt)
        self.U, self.s, self.Vt = U, s, Vt

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _fix_signs(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Singular vectors are only defined up to sign (uᵢ, vᵢ) -> (-uᵢ, -vᵢ).

        Pick the sign that makes the largest-magnitude term loading positive, so the
        output is reproducible and the dominant words of a concept read as positive.
        Flipping uᵢ and vᵢ together leaves A = U Σ Vᵀ unchanged.
        """
        idx = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[idx, np.arange(U.shape[1])])
        signs[signs == 0] = 1.0
        return U * signs, Vt * signs[:, None]

    @property
    def shape(self) -> tuple[int, int]:
        return self.A.shape

    @property
    def rank(self) -> int:
        tol = self.s.max(initial=0) * max(self.A.shape) * np.finfo(float).eps
        return int(np.sum(self.s > tol))

    @property
    def max_k(self) -> int:
        return max(1, self.rank)

    def _k(self, k: int) -> int:
        return int(np.clip(k, 1, len(self.s)))

    # ------------------------------------------------------------ factors
    def U_k(self, k: int) -> np.ndarray:
        return self.U[:, : self._k(k)]

    def S_k(self, k: int) -> np.ndarray:
        return np.diag(self.s[: self._k(k)])

    def Vt_k(self, k: int) -> np.ndarray:
        return self.Vt[: self._k(k)]

    def doc_coords(self, k: int) -> np.ndarray:
        """n x k: document j is row j of V_k Σ_k (equivalently U_kᵀ aⱼ)."""
        k = self._k(k)
        return self.Vt[:k].T * self.s[:k]

    def term_coords(self, k: int) -> np.ndarray:
        """m x k: term i is row i of U_k Σ_k."""
        k = self._k(k)
        return self.U[:, :k] * self.s[:k]

    def reconstruct(self, k: int) -> np.ndarray:
        """A_k = U_k Σ_k V_kᵀ, the best rank-k approximation of A."""
        k = self._k(k)
        return (self.U[:, :k] * self.s[:k]) @ self.Vt[:k]

    def layer(self, i: int) -> np.ndarray:
        """The i-th rank-1 layer σᵢ uᵢ vᵢᵀ (0-based)."""
        return self.s[i] * np.outer(self.U[:, i], self.Vt[i])

    # ------------------------------------------------------- linear algebra checks
    def checks(self) -> dict:
        r = len(self.s)
        I = np.eye(r)
        return {
            "UtU_minus_I": float(np.linalg.norm(self.U.T @ self.U - I)),
            "VtV_minus_I": float(np.linalg.norm(self.Vt @ self.Vt.T - I)),
            "A_minus_USVt": float(np.linalg.norm(self.A - (self.U * self.s) @ self.Vt)),
            "sorted": bool(np.all(np.diff(self.s) <= 1e-12)),
            "nonnegative": bool(np.all(self.s >= 0)),
            # σᵢ² are the eigenvalues of AᵀA (and of AAᵀ)
            "eig_AtA_vs_sigma2": float(
                np.max(np.abs(np.sort(np.linalg.eigvalsh(self.A.T @ self.A))[::-1][:r] - self.s**2))
            ),
        }

    def energy(self) -> np.ndarray:
        """Fraction of ||A||_F² captured by each concept: σᵢ² / Σⱼ σⱼ²."""
        e = self.s**2
        return e / e.sum() if e.sum() > 0 else e

    def eckart_young(self, k: int) -> dict:
        """Check ||A − A_k||_F = sqrt(σ_{k+1}² + … + σ_r²) and ||A − A_k||₂ = σ_{k+1}."""
        k = self._k(k)
        E = self.A - self.reconstruct(k)
        tail = self.s[k:]
        total = float(np.linalg.norm(self.A))
        frob = float(np.linalg.norm(E))
        return {
            "k": k,
            "frobenius_error": frob,
            "frobenius_theory": float(np.sqrt(np.sum(tail**2))),
            "spectral_error": float(np.linalg.norm(E, 2)) if E.size else 0.0,
            "spectral_theory": float(tail[0]) if tail.size else 0.0,
            "relative_error": frob / total if total > 0 else 0.0,
            "energy_kept": float(np.sum(self.s[:k] ** 2) / np.sum(self.s**2)) if total > 0 else 1.0,
            "norm_A": total,
        }

    def random_subspace_error(self, k: int, trials: int = 20, seed: int = 0) -> np.ndarray:
        """||A − QQᵀA||_F for random k-dimensional subspaces Q (always ≥ the SVD error)."""
        k = self._k(k)
        rng = np.random.default_rng(seed)
        errs = []
        for _ in range(trials):
            Q, _ = np.linalg.qr(rng.standard_normal((self.A.shape[0], k)))
            errs.append(np.linalg.norm(self.A - Q @ (Q.T @ self.A)))
        return np.array(errs)

    # -------------------------------------------------------------- concepts
    def concept(self, i: int, n_terms: int = 8, n_docs: int = 3) -> Concept:
        u = self.U[:, i]
        v = self.Vt[i]
        order = np.argsort(u)
        pos = [(str(self.terms[j]), float(u[j])) for j in order[::-1][:n_terms] if u[j] > 0]
        neg = [(str(self.terms[j]), float(u[j])) for j in order[:n_terms] if u[j] < 0]
        dorder = np.argsort(v)
        top = [(int(j), float(v[j])) for j in dorder[::-1][:n_docs]]
        bottom = [(int(j), float(v[j])) for j in dorder[:n_docs]]
        return Concept(i, float(self.s[i]), pos, neg, top, bottom)

    # ----------------------------------------------------------- similarity
    def fold_in(self, q: np.ndarray, k: int) -> np.ndarray:
        """q̂ = Σ_k⁻¹ U_kᵀ q: where a new document would sit as a row of V_k."""
        k = self._k(k)
        s = self.s[:k].copy()
        s[s == 0] = np.inf
        return (self.U[:, :k].T @ q) / s

    def query_coords(self, q: np.ndarray, k: int) -> np.ndarray:
        """Σ_k q̂ = U_kᵀ q: the query in the same coordinates as doc_coords()."""
        return self.U[:, : self._k(k)].T @ q

    def raw_similarity(self, q: np.ndarray) -> np.ndarray:
        """cos(q, aⱼ) for every document column aⱼ of A."""
        return _cosine_rows(self.A.T, q)

    def latent_similarity(self, q: np.ndarray, k: int) -> np.ndarray:
        """cos(U_kᵀ q, U_kᵀ aⱼ) for every document."""
        return _cosine_rows(self.doc_coords(k), self.query_coords(q, k))

    def doc_similarity(self, k: int | None = None) -> np.ndarray:
        """n x n cosine similarity between documents (raw space if k is None)."""
        X = self.A.T if k is None else self.doc_coords(k)
        return cosine_matrix(X)

    def term_vector(self, term: str, k: int | None = None) -> np.ndarray | None:
        hits = np.flatnonzero(self.terms == term.lower())
        if not hits.size:
            return None
        row = self.A[hits[0]] if k is None else self.reconstruct(k)[hits[0]]
        return row

    def term_correlation(self, t1: str, t2: str, k: int | None = None) -> float | None:
        """Pearson correlation between two term rows across documents, in A or in A_k."""
        a, b = self.term_vector(t1, k), self.term_vector(t2, k)
        if a is None or b is None or np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    def unit_doc_coords(self, k: int) -> np.ndarray:
        """Document coordinates scaled to unit length: the space k-means and cosine work in."""
        D = self.doc_coords(k)
        n = np.linalg.norm(D, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return D / n

    def best_axes(self, k: int, labels=None) -> tuple[int, int]:
        """The pair of concept axes (0-based) that best separates the given labels on a 2-D map."""
        k = self._k(k)
        if k < 2:
            return (0, 0)
        if labels is None or len(set(labels)) < 2:
            return (1, 2) if k >= 3 else (0, 1)
        from itertools import combinations

        from sklearn.metrics import silhouette_score

        D = self.unit_doc_coords(k)
        best, pair = -2.0, (0, 1)
        for i, j in combinations(range(min(k, 8)), 2):
            P = D[:, [i, j]]
            if np.allclose(P, 0):
                continue
            s = silhouette_score(P, labels)
            if s > best:
                best, pair = s, (i, j)
        return pair

    def term_correlation_matrix(self, idx: np.ndarray, k: int | None = None) -> np.ndarray:
        M = self.A[idx] if k is None else self.reconstruct(k)[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(M)
        return np.nan_to_num(C)
