"""Document collections used in the project.

Three built-in corpora, chosen so each one teaches something different:

classic   The 9 paper titles from Deerwester et al. (1990), the paper that
          introduced LSA. Only 12 index terms, so every number in A, U, Σ and V
          fits on screen. It is the cleanest demonstration of synonymy:
          "human" and "user" never co-occur, yet LSA learns they are related.

science   The 24 hand-written science blurbs from the first version of this
          project (AI, space, medicine, finance). Nearly every document uses
          its own vocabulary, which is why its singular values are almost flat.

news      400 real news articles (100 each from World, Sports, Business,
          Sci/Tech) sampled from the AG News corpus. Real, messy text where
          clustering in raw TF-IDF space fails and LSA clearly helps.

A user can also supply their own documents (one per line) via custom_corpus().
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Corpus:
    key: str
    name: str
    description: str
    titles: tuple[str, ...]
    texts: tuple[str, ...]
    labels: tuple[str, ...] | None
    # Defaults that suit this corpus. All can be changed in the app.
    weighting: str = "tfidf"
    default_k: int = 4
    min_df: int = 1
    max_df: float = 1.0
    stop_words: str | None = "english"
    vocabulary: tuple[str, ...] | None = None
    short_ids: tuple[str, ...] | None = None
    example_queries: tuple[str, ...] = ()
    word_pairs: tuple[tuple[str, str], ...] = ()
    source: str = ""
    label_order: tuple[str, ...] = field(default=())
    # Optional pair of titles used as the opening example in the app
    showcase: tuple[str, str] | None = None

    @property
    def n_docs(self) -> int:
        return len(self.texts)

    @property
    def has_labels(self) -> bool:
        return self.labels is not None and len(set(self.labels)) > 1

    @property
    def categories(self) -> tuple[str, ...]:
        if not self.labels:
            return ()
        if self.label_order:
            return self.label_order
        seen: list[str] = []
        for lab in self.labels:
            if lab not in seen:
                seen.append(lab)
        return tuple(seen)

    def ids(self) -> list[str]:
        if self.short_ids:
            return list(self.short_ids)
        return [f"d{i + 1}" for i in range(self.n_docs)]

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for t in self.texts:
            h.update(t.encode("utf-8"))
            h.update(b"\x00")
        return f"{self.key}-{h.hexdigest()[:10]}"


# --------------------------------------------------------------------------
# 1. The classic example from the original LSA paper
# --------------------------------------------------------------------------
# Deerwester, Dumais, Furnas, Landauer & Harshman (1990),
# "Indexing by Latent Semantic Analysis", JASIS 41(6), Table 2.
_CLASSIC = [
    ("c1", "Human machine interface for ABC computer applications"),
    ("c2", "A survey of user opinion of computer system response time"),
    ("c3", "The EPS user interface management system"),
    ("c4", "System and human system engineering testing of EPS"),
    ("c5", "Relation of user perceived response time to error measurement"),
    ("m1", "The generation of random, binary, ordered trees"),
    ("m2", "The intersection graph of paths in trees"),
    ("m3", "Graph minors IV: Widths of trees and well-quasi-ordering"),
    ("m4", "Graph minors: A survey"),
]
# The paper keeps only words that occur in at least two titles.
_CLASSIC_TERMS = (
    "human", "interface", "computer", "user", "system", "response",
    "time", "eps", "survey", "trees", "graph", "minors",
)


def _classic() -> Corpus:
    return Corpus(
        key="classic",
        name="Classic: 9 paper titles (1990)",
        description=(
            "The worked example from Deerwester et al. (1990), the paper that introduced LSA. "
            "Five titles are about human–computer interaction and four about graph theory. "
            "With only 12 terms you can see every number in the matrices."
        ),
        titles=tuple(t for _, t in _CLASSIC),
        texts=tuple(t for _, t in _CLASSIC),
        labels=tuple("Human–computer interaction" if i.startswith("c") else "Graph theory" for i, _ in _CLASSIC),
        weighting="count",
        default_k=2,
        stop_words=None,
        vocabulary=_CLASSIC_TERMS,
        short_ids=tuple(i for i, _ in _CLASSIC),
        example_queries=("human computer interaction", "graph survey", "user response time"),
        word_pairs=(("human", "user"), ("human", "minors"), ("trees", "graph")),
        source="Deerwester et al. (1990), Indexing by Latent Semantic Analysis, Table 2.",
        label_order=("Human–computer interaction", "Graph theory"),
    )


# --------------------------------------------------------------------------
# 2. The original 24-document science corpus
# --------------------------------------------------------------------------
def _science() -> Corpus:
    rows = json.loads((DATA_DIR / "science_24.json").read_text(encoding="utf-8"))
    return Corpus(
        key="science",
        name="Science blurbs: 24 short documents",
        description=(
            "24 one-sentence blurbs written for the first version of this project: "
            "AI, space, medicine and finance. Each document uses mostly its own words, "
            "so there is little shared vocabulary for SVD to exploit."
        ),
        titles=tuple(r["title"] for r in rows),
        texts=tuple(r["text"] for r in rows),
        labels=tuple(r["category"] for r in rows),
        weighting="tfidf",
        default_k=4,
        example_queries=(
            "telescope galaxy stars",
            "deep neural network training",
            "immune therapy for cancer patients",
            "central bank interest rates and inflation",
        ),
        word_pairs=(("cosmic", "space"), ("immune", "treat"), ("learning", "networks")),
        source="Written by the project team.",
        label_order=("Artificial Intelligence", "Space Exploration", "Medical Science", "Finance & Economics"),
    )


# --------------------------------------------------------------------------
# 3. Real news articles (AG News subset)
# --------------------------------------------------------------------------
def _news() -> Corpus:
    with open(DATA_DIR / "news_400.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return Corpus(
        key="news",
        name="News: 400 real articles (AG News)",
        description=(
            "400 short news articles, 100 from each AG News section: World, Sports, "
            "Business and Sci/Tech. Real text with real ambiguity, e.g. a tech-company "
            "takeover filed under Sci/Tech that reads like Business."
        ),
        titles=tuple(r["title"] for r in rows),
        texts=tuple(f"{r['title']}. {r['text']}" for r in rows),
        labels=tuple(r["category"] for r in rows),
        weighting="tfidf",
        default_k=5,
        min_df=2,
        example_queries=(
            "computer virus",
            "stock market",
            "championship coach",
            "election violence",
        ),
        word_pairs=(("airline", "airlines"), ("scored", "wins"), ("profits", "shares")),
        source=(
            "AG News test split (Zhang, Zhao & LeCun, 2015), from AG's corpus of news articles. "
            "100 articles sampled per section (seed 7); HTML entities and wire-agency prefixes removed."
        ),
        label_order=("World", "Sports", "Business", "Sci/Tech"),
        showcase=("Autonomy, Mamma.com Join Desktop Search Ranks", "New Version Of Google Groups Launched"),
    )


CORPORA = {
    "news": _news,
    "classic": _classic,
    "science": _science,
}


def load_corpus(key: str) -> Corpus:
    if key not in CORPORA:
        raise KeyError(f"Unknown corpus '{key}'. Choose from: {', '.join(CORPORA)}")
    return CORPORA[key]()


def custom_corpus(raw_text: str) -> Corpus:
    """Build a corpus from pasted text: one document per non-empty line.

    A line may start with a label followed by ' | ', e.g. 'sports | The match ...'.
    Labels are optional, but needed to score clustering against ground truth.
    """
    titles, texts, labels = [], [], []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        label = None
        if " | " in line:
            label, line = (part.strip() for part in line.split(" | ", 1))
        texts.append(line)
        titles.append(line if len(line) <= 70 else line[:67].rstrip() + "…")
        labels.append(label)
    has_all_labels = len(texts) > 0 and all(lab for lab in labels)
    return Corpus(
        key="custom",
        name="Your own documents",
        description=f"{len(texts)} documents you pasted in.",
        titles=tuple(titles),
        texts=tuple(texts),
        labels=tuple(labels) if has_all_labels else None,
        weighting="tfidf",
        default_k=max(1, min(3, len(texts) - 1)),
        source="Pasted by you.",
    )


SAMPLE_CUSTOM_TEXT = """cooking | Simmer the onions in butter until golden, then add garlic and fresh tomatoes.
cooking | Bake the bread at a high oven temperature so the crust turns crisp and golden.
cooking | Whisk eggs with sugar and butter, fold in flour and bake the cake for forty minutes.
cooking | Season the soup with salt, pepper and fresh herbs and let it simmer slowly.
music | The guitarist tuned the strings before the band started the concert.
music | Our choir rehearsed the melody until every voice was in tune.
music | The drummer kept the rhythm steady while the band played the encore.
music | She composed a melody for piano and strings in a minor key.
football | The striker scored a late goal and the team won the match.
football | The goalkeeper saved a penalty in the final minutes of the match.
football | Fans cheered as the team lifted the trophy after the final whistle.
football | The coach changed the formation and the defenders kept a clean sheet."""
