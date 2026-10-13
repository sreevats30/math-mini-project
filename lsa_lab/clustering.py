"""Step 5: cluster documents and check whether the clusters match the real topics.

Two competitors, both using k-means on unit-length document vectors
(so Euclidean distance behaves like cosine distance):

    raw   each document is its column of A (m dimensions, mostly zeros)
    lsa   each document is its row of V_k Σ_k (k dimensions, dense)

k-means depends on its random start, so every score is reported as the mean and
standard deviation over several seeds instead of one lucky run.

Scores:
    ARI  adjusted Rand index      0 = random labelling, 1 = identical to the true topics
    NMI  normalised mutual info   0 = independent,      1 = identical
    silhouette (cosine)           how well separated the clusters are *in their own space*.
                                  It is not comparable across spaces of different dimension,
                                  so we report it for information only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

from .model import LSA

SEEDS = tuple(range(10))


def raw_features(lsa: LSA):
    """Documents as unit-length rows of Aᵀ, kept sparse (most entries are zero, and k-means is faster)."""
    return sp.csr_matrix(normalize(lsa.A.T))


def latent_features(lsa: LSA, k: int) -> np.ndarray:
    return normalize(lsa.doc_coords(k))


def kmeans(X: np.ndarray, n_clusters: int, seed: int = 0, n_init: int = 10) -> np.ndarray:
    n_clusters = int(min(n_clusters, X.shape[0]))
    return KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit_predict(X)


def scores(y_true, y_pred) -> dict:
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
    }


def silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    if len(set(labels)) < 2 or len(set(labels)) >= X.shape[0]:
        return None
    return float(silhouette_score(X, labels, metric="cosine"))


@dataclass
class Comparison:
    n_clusters: int
    k: int
    seeds: tuple[int, ...]
    raw_ari: np.ndarray
    raw_nmi: np.ndarray
    lsa_ari: np.ndarray
    lsa_nmi: np.ndarray
    raw_silhouette: float | None
    lsa_silhouette: float | None
    labels_raw: np.ndarray      # from the first seed
    labels_lsa: np.ndarray      # from the first seed

    def summary(self) -> dict:
        def ms(x):
            return {"mean": float(np.mean(x)), "std": float(np.std(x))}

        return {
            "n_clusters": self.n_clusters,
            "k": self.k,
            "seeds": list(self.seeds),
            "raw": {"ARI": ms(self.raw_ari), "NMI": ms(self.raw_nmi), "silhouette_cosine": self.raw_silhouette},
            "lsa": {"ARI": ms(self.lsa_ari), "NMI": ms(self.lsa_nmi), "silhouette_cosine": self.lsa_silhouette},
        }


def evaluate_space(X: np.ndarray, y_true, n_clusters: int, seeds=SEEDS) -> dict:
    """k-means on X for every seed; ARI/NMI against y_true (if given)."""
    ari, nmi, first = [], [], None
    for s in seeds:
        lab = kmeans(X, n_clusters, s)
        if first is None:
            first = lab
        if y_true is not None:
            sc = scores(y_true, lab)
            ari.append(sc["ARI"])
            nmi.append(sc["NMI"])
    return {"ARI": np.array(ari), "NMI": np.array(nmi), "labels": first, "silhouette": silhouette(X, first)}


def compare(lsa: LSA, y_true, k: int, n_clusters: int, seeds=SEEDS, raw: dict | None = None) -> Comparison:
    """Raw TF-IDF space vs. k-dimensional concept space. Pass `raw` to reuse a cached raw run."""
    raw = raw or evaluate_space(raw_features(lsa), y_true, n_clusters, seeds)
    lat = evaluate_space(latent_features(lsa, k), y_true, n_clusters, seeds)
    return Comparison(
        n_clusters=n_clusters,
        k=k,
        seeds=tuple(seeds),
        raw_ari=raw["ARI"],
        raw_nmi=raw["NMI"],
        lsa_ari=lat["ARI"],
        lsa_nmi=lat["NMI"],
        raw_silhouette=raw["silhouette"],
        lsa_silhouette=lat["silhouette"],
        labels_raw=raw["labels"],
        labels_lsa=lat["labels"],
    )


def k_sweep(lsa: LSA, y_true, n_clusters: int, ks, seeds=SEEDS, n_init: int = 10) -> dict:
    """Clustering quality (ARI) as a function of the number of concepts k."""
    ks = [k for k in ks if 1 <= k <= lsa.max_k]
    Xr = raw_features(lsa)
    raw = [scores(y_true, kmeans(Xr, n_clusters, s, n_init))["ARI"] for s in seeds]
    means, stds = [], []
    for k in ks:
        Xl = latent_features(lsa, k)
        vals = [scores(y_true, kmeans(Xl, n_clusters, s, n_init))["ARI"] for s in seeds]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return {
        "k": list(ks),
        "lsa_mean": means,
        "lsa_std": stds,
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
    }


def match_clusters(y_true, y_pred, categories) -> dict[int, str]:
    """Name each cluster after the true category it overlaps most (Hungarian matching)."""
    cats = list(categories)
    clusters = sorted(set(int(c) for c in y_pred))
    C = np.zeros((len(clusters), len(cats)))
    for c, t in zip(y_pred, y_true):
        C[clusters.index(int(c)), cats.index(t)] += 1
    rows, cols = linear_sum_assignment(-C)
    names = {clusters[r]: cats[c] for r, c in zip(rows, cols)}
    for c in clusters:
        names.setdefault(c, f"Cluster {c + 1}")
    return names


def confusion(y_true, y_pred, categories, names: dict[int, str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Rows: true category. Columns: found cluster, ordered to line up with the rows."""
    cats = list(categories)
    order = sorted(names, key=lambda c: cats.index(names[c]) if names[c] in cats else len(cats) + c)
    M = np.zeros((len(cats), len(order)), dtype=int)
    for c, t in zip(y_pred, y_true):
        M[cats.index(t), order.index(int(c))] += 1
    return M, cats, [names[c] for c in order]


def cluster_keywords(lsa: LSA, k: int, labels: np.ndarray, n: int = 8) -> dict[int, list[str]]:
    """Why do these documents belong together? Top terms of each cluster's centre.

    The centre of a cluster in concept space, mapped back to words with U_k,
    is the average of its documents' columns in A_k.
    """
    Uk = lsa.U_k(k)
    D = lsa.doc_coords(k)
    out = {}
    for c in sorted(set(int(x) for x in labels)):
        centre = Uk @ D[labels == c].mean(axis=0)
        out[c] = [str(lsa.terms[i]) for i in np.argsort(centre)[::-1][:n]]
    return out
