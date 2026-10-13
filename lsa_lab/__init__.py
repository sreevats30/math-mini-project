"""LSA Lab: Latent Semantic Analysis (TF-IDF + SVD) for clustering documents by topic.

The package is split by pipeline step so each file maps to one idea:

    corpora     the document collections we analyse
    text        documents -> term-document matrix A (counts or TF-IDF)
    model       A = U Σ Vᵀ, truncation to rank k, folding-in, similarities
    clustering  k-means in raw TF-IDF space vs. LSA concept space, evaluation
    figures     static figures for the report and the slide deck
    deck        builds the PowerPoint presentation
"""

from .corpora import Corpus, CORPORA, load_corpus, custom_corpus
from .text import TermDocumentMatrix, build_matrix
from .model import LSA
from .pipeline import Experiment, run_experiment

__all__ = [
    "Corpus",
    "CORPORA",
    "load_corpus",
    "custom_corpus",
    "TermDocumentMatrix",
    "build_matrix",
    "LSA",
    "Experiment",
    "run_experiment",
]
