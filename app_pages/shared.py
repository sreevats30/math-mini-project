"""Shared pieces of the learning app: state, caching, layout components and chart styling."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lsa_lab import clustering as cl
from lsa_lab.corpora import CORPORA, custom_corpus, load_corpus
from lsa_lab.pipeline import Experiment, run_experiment
from lsa_lab.style import (
    A_COLOR,
    DIVERGING,
    INK,
    INK_FAINT,
    INK_SOFT,
    RULE,
    S_COLOR,
    SEQUENTIAL,
    U_COLOR,
    V_COLOR,
    category_style,
)

# --------------------------------------------------------------------------- chapters
CHAPTERS = [
    ("app_pages/c1_words.py", "Documents become word counts"),
    ("app_pages/c2_matrix.py", "Build the term–document matrix"),
    ("app_pages/c3_svd.py", "Factorise A = UΣVᵀ"),
    ("app_pages/c4_choose_k.py", "Keep the strongest k concepts"),
    ("app_pages/c5_concepts.py", "Read the concepts"),
    ("app_pages/c6_clusters.py", "Cluster the ideas"),
    ("app_pages/c7_search.py", "Search by meaning"),
]
NAV_TITLES = ["Word counts", "The matrix A", "SVD: A = UΣVᵀ", "Choosing k", "Reading concepts",
              "Clustering", "Search by meaning"]

CORPUS_LABELS = {
    "news": "News: 400 real articles",
    "classic": "Classic: 9 paper titles (1990)",
    "science": "Science blurbs: 24 short texts",
    "custom": "Your own documents",
}
WEIGHTING_LABELS = {"tfidf": "TF-IDF", "count": "Raw counts"}


# --------------------------------------------------------------------------- state
def init_state():
    ss = st.session_state
    if "corpus_key" not in ss:
        ss.corpus_key = "news"
        ss.custom_text = ""
        _apply_corpus_defaults()


def _apply_corpus_defaults():
    ss = st.session_state
    c = get_corpus(ss.corpus_key, ss.get("custom_text", ""))
    ss.weighting = c.weighting
    ss.k = c.default_k
    for key in [key for key in ss.keys() if str(key).startswith("k_slider_")]:
        del ss[key]


def on_corpus_change():
    _apply_corpus_defaults()


def keep_weighting():
    """Clicking the selected option of a segmented control clears it; keep the last choice instead."""
    ss = st.session_state
    if ss.weighting is None:
        ss.weighting = ss.get("_weighting_prev", "tfidf")
    ss._weighting_prev = ss.weighting


def corpus_options() -> list[str]:
    keys = list(CORPORA)
    if st.session_state.get("custom_text"):
        keys.append("custom")
    return keys


@st.cache_resource(show_spinner=False)
def get_corpus(key: str, custom_text: str = ""):
    if key == "custom":
        return custom_corpus(custom_text)
    return load_corpus(key)


@st.cache_resource(show_spinner="Building the term–document matrix and its SVD…")
def get_experiment(key: str, weighting: str, custom_text: str = "") -> Experiment:
    return run_experiment(get_corpus(key, custom_text), weighting)


@dataclass
class Ctx:
    ex: Experiment
    k: int
    key: str
    weighting: str
    custom_text: str

    @property
    def corpus(self):
        return self.ex.corpus

    @property
    def lsa(self):
        return self.ex.lsa

    @property
    def tdm(self):
        return self.ex.tdm

    @property
    def cache_args(self) -> tuple:
        return (self.key, self.weighting, self.custom_text)


def ctx() -> Ctx:
    ss = st.session_state
    ex = get_experiment(ss.corpus_key, ss.weighting, ss.get("custom_text", "") if ss.corpus_key == "custom" else "")
    k = int(np.clip(ss.k, 1, max_k(ex)))
    return Ctx(ex, k, ss.corpus_key, ss.weighting, ss.get("custom_text", "") if ss.corpus_key == "custom" else "")


def max_k(ex: Experiment) -> int:
    return int(max(1, min(ex.lsa.max_k, 50)))


# --------------------------------------------------------------------------- cached analysis
@st.cache_data(show_spinner="Clustering the raw TF-IDF vectors (10 runs)…", persist="disk")
def raw_eval(key: str, weighting: str, custom_text: str, n_clusters: int) -> dict:
    ex = get_experiment(key, weighting, custom_text)
    return cl.evaluate_space(cl.raw_features(ex.lsa), ex.labels, n_clusters)


@st.cache_data(show_spinner="Clustering in concept space (10 runs)…")
def lsa_eval(key: str, weighting: str, custom_text: str, k: int, n_clusters: int) -> dict:
    ex = get_experiment(key, weighting, custom_text)
    return cl.evaluate_space(cl.latent_features(ex.lsa, k), ex.labels, n_clusters)


from lsa_lab.report import SWEEP_KS, ROOT as _ROOT  # noqa: E402


def _precomputed_sweep(key: str, weighting: str, custom_text: str, n_clusters: int) -> dict | None:
    """The default news sweep is saved by `python main.py report`; reuse it so the page opens instantly."""
    import json

    path = _ROOT / "results" / "metrics.json"
    if custom_text or key != "news" or n_clusters != 4 or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["corpora"]["news"]["weighting"] != weighting:
            return None
        sweep = data["news_k_sweep"]
        return sweep if sweep["k"] == [k for k in SWEEP_KS if k <= 399] else None
    except (KeyError, ValueError):
        return None


@st.cache_data(show_spinner="Trying every k: 10 clustering runs each. This takes a few seconds, once.",
               persist="disk")
def k_sweep(key: str, weighting: str, custom_text: str, n_clusters: int) -> dict:
    pre = _precomputed_sweep(key, weighting, custom_text, n_clusters)
    if pre is not None:
        return pre
    ex = get_experiment(key, weighting, custom_text)
    return cl.k_sweep(ex.lsa, ex.labels, n_clusters, SWEEP_KS)


# --------------------------------------------------------------------------- page chrome
def css():
    st.markdown(
        f"""
<style>
:root {{
  --ink: {INK}; --ink-soft: {INK_SOFT}; --ink-faint: {INK_FAINT}; --rule: {RULE};
  --u: {U_COLOR}; --s: {S_COLOR}; --v: {V_COLOR}; --a: {A_COLOR};
  --paper: #FAFBFC; --panel: #EEF2F5;
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h1 {{ font-weight: 600 !important; }}
.lab-count {{ color: var(--ink-soft); font-size: 0.92rem; margin-bottom: 0.35rem; }}
.lab-progress {{ display: flex; gap: 4px; margin: 0 0 1.1rem 0; max-width: 360px; }}
.lab-progress span {{ flex: 1; height: 5px; border-radius: 3px; background: var(--rule); }}
.lab-progress span.done {{ background: var(--u); opacity: 0.45; }}
.lab-progress span.here {{ background: var(--u); }}
.lab-lede {{ font-size: 1.12rem; line-height: 1.6; color: var(--ink); max-width: 46rem; margin-bottom: 1.2rem; }}
.lab-prose {{ max-width: 46rem; line-height: 1.65; }}
.lab-key {{ border-left: 4px solid var(--u); background: #E6F2F2; padding: 0.85rem 1.1rem; border-radius: 0 0.45rem 0.45rem 0;
            margin: 0.6rem 0 1.2rem 0; max-width: 50rem; }}
.lab-key b.lab-tag, .lab-try b.lab-tag, .lab-warn b.lab-tag {{ font-weight: 600; margin-right: 0.35rem; }}
.lab-key p, .lab-try p {{ margin: 0; font-family: 'Literata', serif; font-size: 1.05rem; line-height: 1.6; }}
.lab-try {{ border-left: 4px solid var(--s); background: #F7EFDD; padding: 0.75rem 1.1rem; border-radius: 0 0.45rem 0.45rem 0;
            margin: 0.6rem 0 1.2rem 0; max-width: 50rem; }}
.lab-warn {{ border-left: 4px solid #B3261E; background: #FBEAE8; padding: 0.75rem 1.1rem; border-radius: 0 0.45rem 0.45rem 0;
             margin: 0.6rem 0 1.2rem 0; max-width: 50rem; }}
.lab-measure {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0; border: 1px solid var(--rule);
                border-radius: 0.45rem; background: white; margin: 0.4rem 0 1.2rem 0; }}
.lab-measure > div {{ padding: 0.8rem 1rem; border-right: 1px solid var(--rule); }}
.lab-measure > div:last-child {{ border-right: none; }}
.lab-measure .v {{ font-family: 'Literata', serif; font-size: 1.75rem; font-weight: 600; line-height: 1.2; color: var(--ink); }}
.lab-measure .l {{ font-size: 0.88rem; color: var(--ink-soft); line-height: 1.35; }}
.lab-doc {{ background: white; border: 1px solid var(--rule); border-radius: 0.45rem; padding: 0.9rem 1.1rem; height: 100%; }}
.lab-doc .meta {{ font-size: 0.85rem; color: var(--ink-soft); margin-bottom: 0.3rem; }}
.lab-doc .body {{ font-family: 'Literata', serif; font-size: 1.04rem; line-height: 1.65; }}
.tok-kept {{ background: #DCEEEE; border-radius: 3px; padding: 0 2px; }}
.tok-shared {{ background: #F3DDB0; border-radius: 3px; padding: 0 2px; }}
.tok-stop {{ color: var(--ink-faint); }}
.svd-wrap {{ display: flex; align-items: flex-end; gap: 14px; flex-wrap: wrap; margin: 0.6rem 0 0.4rem 0; }}
.svd-item {{ display: flex; flex-direction: column; align-items: center; gap: 6px; }}
.svd-box {{ border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white;
            font-family: 'Literata', serif; font-weight: 600; font-size: 1.15rem; }}
.svd-dim {{ font-size: 0.82rem; color: var(--ink-soft); white-space: nowrap; }}
.svd-box .sb {{ font-size: 0.7em; vertical-align: -0.35em; margin-left: 1px; }}
.svd-box .sp {{ font-size: 0.7em; vertical-align: 0.55em; margin-left: 1px; }}
.svd-op {{ font-family: 'Literata', serif; font-size: 1.6rem; color: var(--ink-soft); padding-bottom: 34px; }}
.lab-route ol {{ padding-left: 1.3rem; }}
.lab-route li {{ margin: 0.25rem 0; }}
.lab-small {{ font-size: 0.88rem; color: var(--ink-soft); }}
[data-testid="stSidebarNav"] li a span {{ font-size: 0.95rem; }}
</style>
""",
        unsafe_allow_html=True,
    )


def chapter_header(n: int, title: str, lede: str):
    total = len(CHAPTERS)
    segs = "".join(
        f'<span class="{"here" if i == n else "done" if i < n else ""}"></span>' for i in range(1, total + 1)
    )
    st.markdown(f'<div class="lab-count">Chapter {n} of {total}</div><div class="lab-progress">{segs}</div>',
                unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(f'<div class="lab-lede">{lede}</div>', unsafe_allow_html=True)


def chapter_footer(n: int):
    st.divider()
    left, _, right = st.columns([1, 0.2, 1])
    if n > 1:
        path, title = CHAPTERS[n - 2]
        left.page_link(path, label=f"Back: {n - 1}. {title}", icon=":material/arrow_back:")
    else:
        left.page_link("app_pages/home.py", label="Back: the big idea", icon=":material/arrow_back:")
    if n < len(CHAPTERS):
        path, title = CHAPTERS[n]
        right.page_link(path, label=f"Next: {n + 1}. {title}", icon=":material/arrow_forward:")
    else:
        right.page_link("app_pages/formulas.py", label="Next: formula sheet and viva questions",
                        icon=":material/arrow_forward:")


def _md_inline(text: str) -> str:
    """Tiny markdown for callouts: **bold**, *italic*, `code`."""
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t


def key_idea(text: str):
    st.markdown(f'<div class="lab-key"><p><b class="lab-tag">Key idea.</b>{_md_inline(text)}</p></div>',
                unsafe_allow_html=True)


def try_this(text: str):
    st.markdown(f'<div class="lab-try"><p><b class="lab-tag">Try this.</b>{_md_inline(text)}</p></div>',
                unsafe_allow_html=True)


def warn(text: str):
    st.markdown(f'<div class="lab-warn"><b class="lab-tag">Heads up.</b>{_md_inline(text)}</div>',
                unsafe_allow_html=True)


def measures(items: list[tuple[str, str]]):
    cells = "".join(f'<div><div class="v">{html.escape(v)}</div><div class="l">{html.escape(l)}</div></div>'
                    for v, l in items)
    st.markdown(f'<div class="lab-measure">{cells}</div>', unsafe_allow_html=True)


def check_yourself(key: str, question: str, options: list[str], answer: int, explanation: str):
    with st.expander("Check yourself", icon=":material/quiz:"):
        st.markdown(f"**{question}**")
        choice = st.radio(question, options, index=None, key=f"quiz_{key}", label_visibility="collapsed")
        if choice is not None:
            if options.index(choice) == answer:
                st.success(f"Right. {explanation}", icon=":material/check_circle:")
            else:
                st.error(f"Not quite. {explanation}", icon=":material/cancel:")


def suggest_classic(reason: str):
    """On big corpora, offer the 9-title example where every number fits on screen."""
    if st.session_state.corpus_key == "classic":
        return
    c1, c2 = st.columns([4, 1.3], vertical_alignment="center")
    c1.markdown(f'<div class="lab-small">{_md_inline(reason)}</div>', unsafe_allow_html=True)
    c2.button("Use the 9-title example", key=f"to_classic_{abs(hash(reason)) % 10**8}",
              icon=":material/swap_horiz:", on_click=switch_corpus, args=("classic",))


def switch_corpus(key: str):
    st.session_state.corpus_key = key
    _apply_corpus_defaults()


# --------------------------------------------------------------------------- text helpers
def highlight_tokens(text: str, kept: set[str], shared: set[str] | None = None) -> str:
    """Wrap each word: shared words, kept vocabulary words, and dropped words look different."""
    out = []
    for piece in re.split(r"(\b[a-zA-Z][a-zA-Z]+\b)", text):
        w = piece.lower()
        if re.fullmatch(r"[a-zA-Z][a-zA-Z]+", piece):
            if shared and w in shared:
                out.append(f'<span class="tok-shared">{html.escape(piece)}</span>')
            elif w in kept:
                out.append(f'<span class="tok-kept">{html.escape(piece)}</span>')
            else:
                out.append(f'<span class="tok-stop">{html.escape(piece)}</span>')
        else:
            out.append(html.escape(piece))
    return "".join(out)


def doc_card(title: str, meta: str, body_html: str):
    st.markdown(
        f'<div class="lab-doc"><div class="meta">{html.escape(meta)}</div><div class="body">{body_html}</div></div>',
        unsafe_allow_html=True,
    )


def doc_label(c: Ctx, j: int) -> str:
    t = c.corpus.titles[j]
    prefix = c.corpus.ids()[j]
    return f"{prefix}: {t}" if c.corpus.short_ids else t


# --------------------------------------------------------------------------- SVD diagram
def svd_diagram(m: int, n: int, k: int, truncated: bool = True):
    """The factor shapes drawn to (compressed) scale with the real dimensions."""

    def size(x, lo=26, hi=190):
        return int(np.clip(lo + (hi - lo) * np.log1p(x) / np.log1p(2000), lo, hi))

    h_m, w_n, s_k = size(m), size(n), size(k, 22, 120)
    sub = "k" if truncated else "r"
    parts = [
        ("A", A_COLOR, w_n, h_m, f"{m} terms × {n} docs"),
        ("=", None, 0, 0, ""),
        (f'U<span class="sb">{sub}</span>', U_COLOR, s_k, h_m, f"{m} × {k}"),
        ("·", None, 0, 0, ""),
        (f'Σ<span class="sb">{sub}</span>', S_COLOR, s_k, s_k, f"{k} × {k}"),
        ("·", None, 0, 0, ""),
        (f'V<span class="sb">{sub}</span><span class="sp">T</span>', V_COLOR, w_n, s_k, f"{k} × {n}"),
    ]
    if truncated:
        parts[1] = ("≈", None, 0, 0, "")
    html_parts = []
    for label, color, w, h, dim in parts:
        if color is None:
            html_parts.append(f'<div class="svd-op">{label}</div>')
        else:
            html_parts.append(
                f'<div class="svd-item"><div class="svd-box" style="width:{w}px;height:{h}px;background:{color}">'
                f"{label}</div><div class=\"svd-dim\">{dim}</div></div>"
            )
    st.markdown(f'<div class="svd-wrap">{"".join(html_parts)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- charts
def style_fig(fig: go.Figure, height: int = 420, legend: bool = True, title: str | None = None) -> go.Figure:
    top = 48 if title else 16
    if getattr(fig.layout.xaxis, "side", None) == "top":
        top += 56
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=top, b=8),
        title=dict(text=title, x=0, xanchor="left", font=dict(family="Literata, serif", size=17, color=INK)) if title else None,
        font=dict(family="Lexend, sans-serif", size=13, color=INK_SOFT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK)),
        hoverlabel=dict(bgcolor="white", bordercolor=RULE, font=dict(family="Lexend, sans-serif", color=INK, size=13)),
    )
    fig.update_xaxes(gridcolor="#E6EBF0", zeroline=False, linecolor=RULE, ticks="", title_font=dict(color=INK_SOFT))
    fig.update_yaxes(gridcolor="#E6EBF0", zeroline=False, linecolor=RULE, ticks="", title_font=dict(color=INK_SOFT))
    return fig


def show(fig: go.Figure, key: str | None = None):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def seq_scale():
    return [[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]


def div_scale():
    return [[i / (len(DIVERGING) - 1), c] for i, c in enumerate(DIVERGING)]


def heatmap(M, rows, cols, signed=False, text=True, fmt=".2f", height=None, zmax=None, hover="value",
            col_names=None, row_names=None, gap=1.5) -> go.Figure:
    M = np.asarray(M, dtype=float)
    zmax = zmax or (float(np.max(np.abs(M))) or 1.0)
    t = None
    if text:
        def _fmt(v):
            if abs(v) < 0.005:
                return ""
            out = f"{v:{fmt}}"
            return out.lstrip("-") if float(out) == 0 else out

        t = [[_fmt(v) for v in row] for row in M]
    col_names = list(col_names) if col_names is not None else [str(x) for x in cols]
    row_names = list(row_names) if row_names is not None else [str(x) for x in rows]
    custom = np.array([[[rn, cn] for cn in col_names] for rn in row_names], dtype=object)
    fig = go.Figure(
        go.Heatmap(
            z=M,
            x=list(cols),
            y=list(rows),
            colorscale=div_scale() if signed else seq_scale(),
            zmin=-zmax if signed else 0,
            zmax=zmax,
            text=t,
            texttemplate="%{text}" if text else None,
            textfont=dict(size=11, family="Lexend, sans-serif"),
            xgap=gap,
            ygap=gap,
            customdata=custom,
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>" + hover + " = %{z:.3f}<extra></extra>",
            colorbar=dict(thickness=10, outlinewidth=0, len=0.8),
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(side="top", showgrid=False, type="category")
    return style_fig(fig, height=height or max(260, 24 * len(rows) + 90), legend=False)


def category_colors(c: Ctx) -> dict:
    return category_style(c.corpus.categories) if c.corpus.labels else {}


def sample_docs(c: Ctx, per_label: int = 6, cap: int = 24) -> list[int]:
    """A small, label-balanced, reproducible set of documents for matrix views."""
    n = c.tdm.n_docs
    if n <= cap:
        return list(range(n))
    if c.corpus.labels:
        y = np.array(c.corpus.labels)
        out = []
        for cat in c.corpus.categories:
            idx = np.flatnonzero(y == cat)
            # documents with the most indexed words are the most readable
            sizes = (c.tdm.counts[:, idx] > 0).sum(axis=0)
            out.extend(idx[np.argsort(-sizes, kind="stable")[:per_label]].tolist())
        return out[:cap]
    return list(range(cap))


def doc_short(c: Ctx, docs: list[int]) -> list[str]:
    """Short unique column labels: c1…m4 for the classic set, 'Wor 3' style for labelled sets."""
    if c.corpus.short_ids:
        ids = c.corpus.ids()
        return [ids[j] for j in docs]
    if not c.corpus.labels:
        return [f"d{j + 1}" for j in docs]
    seen: dict[str, int] = {}
    out = []
    for j in docs:
        lab = c.corpus.labels[j]
        seen[lab] = seen.get(lab, 0) + 1
        out.append(f"{lab[:3]} {seen[lab]}")
    return out


def top_terms_for(c: Ctx, docs: list[int], n: int = 24) -> list[int]:
    """Rows for matrix views. With labels: the most characteristic words of each topic, grouped by topic."""
    if c.tdm.n_terms <= n:
        return list(range(c.tdm.n_terms))
    A = c.tdm.A[:, docs]
    if not c.corpus.labels:
        return np.argsort(-A.sum(axis=1))[:n].tolist()
    y = np.array([c.corpus.labels[j] for j in docs])
    cats = [g for g in c.corpus.categories if g in set(y)]
    per = max(1, n // len(cats))
    chosen: list[int] = []
    for g in cats:
        inside = A[:, y == g].mean(axis=1)
        outside = A[:, y != g].mean(axis=1) if np.any(y != g) else 0
        picked = 0
        for t in np.argsort(-(inside - outside)):
            if int(t) not in chosen:
                chosen.append(int(t))
                picked += 1
            if picked == per:
                break
    return chosen[:n]


SUB = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")


def sub(x) -> str:
    """Unicode subscript for numbers, e.g. sub(12) -> '₁₂'."""
    return str(x).translate(SUB)
