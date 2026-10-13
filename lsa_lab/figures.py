"""Static figures for the slide deck and the README (matplotlib).

Every figure is computed from the pipeline at build time, so the numbers on the
slides always match the code.   python main.py figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import clustering as cl  # noqa: E402
from .pipeline import Experiment  # noqa: E402
from .style import (  # noqa: E402
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

ASSETS = Path(__file__).resolve().parent / "assets"
SEQ = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
DIV = LinearSegmentedColormap.from_list("div", DIVERGING)


def _setup():
    for f in ASSETS.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update(
        {
            "font.family": ["Lexend", "DejaVu Sans"],
            "font.size": 11,
            "mathtext.fontset": "cm",
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "grid.color": "#E6EBF0",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )


def _heat(ax, M, rows, cols, cmap, norm=None, fmt="{:.0f}", vmin=None, vmax=None, fontsize=9.5, hide_zero=True):
    im = ax.imshow(M, cmap=cmap, norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    rng = (np.max(np.abs(M)) or 1.0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if hide_zero and abs(v) < 0.005:
                continue
            light = abs(v) / rng > 0.55
            label = fmt.format(v)
            if label.lstrip("-").strip("0.") == "":
                label = label.lstrip("-")
            ax.text(j, i, label, ha="center", va="center", fontsize=fontsize, color="white" if light else INK)
    return im


# --------------------------------------------------------------------------- classic
def classic_matrix(ex: Experiment, path: Path):
    A = ex.tdm.counts
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    _heat(ax, A, list(ex.tdm.terms), ex.corpus.ids(), SEQ, vmin=0, vmax=2.6)
    ax.axvline(4.5, color=INK_SOFT, linewidth=1.2)
    ax.text(2, -1.6, "Human–computer interaction", ha="center", fontsize=10, color=INK_SOFT)
    ax.text(6.5, -1.6, "Graph theory", ha="center", fontsize=10, color=INK_SOFT)
    fig.savefig(path)
    plt.close(fig)


def classic_layers(ex: Experiment, path: Path, k: int = 2):
    lsa = ex.lsa
    A, Ak = lsa.A, lsa.reconstruct(k)
    lim = float(np.max(np.abs(A)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), gridspec_kw={"wspace": 0.06})
    titles = ["A: word counts", r"$A_2 = \sigma_1 u_1 v_1^{\top} + \sigma_2 u_2 v_2^{\top}$"]
    for ax, M, t, fmt in zip(axes, [A, Ak], titles, ["{:.0f}", "{:.1f}"]):
        _heat(ax, M, list(ex.tdm.terms), ex.corpus.ids(), DIV, norm=norm, fmt=fmt, fontsize=10)
        ax.set_title(t, pad=30, fontsize=14)
        ax.axvline(4.5, color=INK_SOFT, linewidth=1)
    axes[1].set_yticklabels([])
    # cells that were 0 in A but are clearly positive in A_k
    for i, j in zip(*np.nonzero((A == 0) & (Ak >= 0.35))):
        axes[1].add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=INK, linewidth=2.0, zorder=5))
    fig.text(0.52, 0.035, "Outlined: the title never uses the word, but after keeping 2 concepts it carries it.",
             fontsize=11, color=INK_SOFT)
    fig.savefig(path)
    plt.close(fig)


def classic_synonymy(ex: Experiment, path: Path, k: int = 2):
    lsa = ex.lsa
    idx = np.arange(len(lsa.terms))
    before = lsa.term_correlation_matrix(idx)
    after = lsa.term_correlation_matrix(idx, k)
    terms = list(lsa.terms)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), gridspec_kw={"wspace": 0.08})
    for ax, M, t in zip(axes, [before, after], ["Correlation of word rows in A", f"… and in A{chr(0x2080 + k)} (rank {k})"]):
        _heat(ax, M, terms, terms, DIV, norm=norm, fmt="{:.1f}", fontsize=7.5, hide_zero=False)
        ax.set_title(t, pad=34)
        ax.tick_params(axis="x", labelrotation=90)
        h, u = terms.index("human"), terms.index("user")
        for (r, c) in [(h, u), (u, h)]:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2.2))
    axes[1].set_yticklabels([])
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- news
def scree(ex: Experiment, path: Path, k: int, n_show: int = 40):
    lsa = ex.lsa
    s = lsa.s
    energy = np.cumsum(lsa.energy())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"wspace": 0.28, "width_ratios": [1.25, 1]})
    i = np.arange(1, n_show + 1)
    a1.bar(i, s[:n_show], color=[S_COLOR if x <= k else "#C9D1DA" for x in i], width=0.72)
    a1.set_xlabel("concept i")
    a1.set_ylabel(r"singular value $\sigma_i$")
    a1.set_title(f"Strength of each concept (first {n_show} of {len(s)})")
    a1.grid(axis="y")
    a1.set_axisbelow(True)
    a1.text(k + 0.8, s[0] * 0.92, f"kept: k = {k}", color=S_COLOR, fontsize=10.5, fontweight="bold")
    r = np.arange(1, len(s) + 1)
    a2.plot(r, energy * 100, color=S_COLOR, linewidth=2)
    a2.scatter([k], [energy[k - 1] * 100], color=S_COLOR, s=40, zorder=3)
    a2.annotate(f"k = {k} keeps {energy[k - 1] * 100:.1f}%\n" + r"of $\|A\|_F^2$", (k, energy[k - 1] * 100),
                xytext=(40, 30), textcoords="offset points", fontsize=10, color=INK_SOFT,
                arrowprops={"arrowstyle": "-", "color": INK_FAINT})
    a2.set_xlabel("number of concepts kept, k")
    a2.set_ylabel("energy kept (%)")
    a2.set_ylim(0, 102)
    a2.set_title(r"Cumulative energy  $\sum_{i\leq k}\sigma_i^2\,/\,\|A\|_F^2$")
    a2.grid()
    a2.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def eckart_young(ex: Experiment, path: Path, k: int, k_max: int = 120):
    lsa = ex.lsa
    ks = np.arange(1, k_max + 1)
    theory = np.array([np.sqrt(np.sum(lsa.s[kk:] ** 2)) for kk in ks])
    sample = np.unique(np.r_[1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 80, 100, 120, k])
    computed = np.array([lsa.eckart_young(kk)["frobenius_error"] for kk in sample])
    rand = np.array([lsa.random_subspace_error(kk, trials=8, seed=1) for kk in sample])
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.fill_between(sample, rand.min(1), rand.max(1), color="#C9D1DA", alpha=0.7, linewidth=0)
    ax.plot(sample, rand.mean(1), color=INK_FAINT, linewidth=1.5)
    ax.plot(ks, theory, color=U_COLOR, linewidth=2)
    ax.scatter(sample, computed, color=V_COLOR, s=26, zorder=3)
    ax.text(sample[-1], rand.max(1)[-1] + 0.45, "random k-dimensional subspaces (8 tries each)", color=INK_SOFT, fontsize=10.5, ha="right", va="bottom")
    ax.text(ks[62], theory[62] + 0.35, r"theory  $\sqrt{\sigma_{k+1}^2+\cdots+\sigma_r^2}$", color=U_COLOR, fontsize=12, ha="left", va="bottom")
    ax.text(ks[30], theory[30] - 0.55, r"dots: measured $\|A-A_k\|_F$", color=V_COLOR, fontsize=11, ha="right", va="top")
    ax.set_xlabel("rank k")
    ax.set_ylabel(r"approximation error  $\|A - B\|_F$")
    ax.grid()
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def concepts(ex: Experiment, path: Path, which=(1, 2, 3, 4), n=6):
    lsa = ex.lsa
    fig, axes = plt.subplots(1, len(which), figsize=(4.0 * len(which), 4.4), gridspec_kw={"wspace": 0.55})
    for ax, i in zip(np.atleast_1d(axes), which):
        c = lsa.concept(i, n)
        items = sorted(c.negative + c.positive, key=lambda x: x[1])
        labels = [t for t, _ in items]
        vals = [w for _, w in items]
        ax.barh(range(len(vals)), vals, color=[DIVERGING[0] if v < 0 else DIVERGING[-1] for v in vals], height=0.68)
        ax.set_yticks(range(len(vals)), labels)
        ax.axvline(0, color=INK_SOFT, linewidth=0.8)
        ax.set_title(f"Concept {i + 1}   σ = {c.sigma:.2f}", fontsize=12)
        ax.set_xlabel(r"loading in $u_{%d}$" % (i + 1))
        lim = max(abs(min(vals)), abs(max(vals))) * 1.1
        ax.set_xlim(-lim, lim)
        t = round(lim * 0.7, 1) or round(lim * 0.7, 2)
        ax.set_xticks([-t, 0, t], [f"{-t:g}", "0", f"{t:g}"])
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def doc_map(ex: Experiment, path: Path, k: int):
    lsa = ex.lsa
    y = ex.labels
    i, j = lsa.best_axes(k, y)
    D = lsa.unit_doc_coords(k)
    st = category_style(ex.corpus.categories)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for cat, (col, _, m) in st.items():
        msk = y == cat
        ax.scatter(D[msk, i], D[msk, j], c=col, marker=m, s=30, alpha=0.85, edgecolors="white", linewidths=0.5, label=cat)
    ax.axhline(0, color=RULE, linewidth=0.8, zorder=0)
    ax.axvline(0, color=RULE, linewidth=0.8, zorder=0)
    ax.set_xlabel(f"concept {i + 1}")
    ax.set_ylabel(f"concept {j + 1}")
    ax.set_title(f"{ex.tdm.n_docs} news articles in concept space (k = {k}, unit length)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.3, markerscale=1.4, fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(path)
    plt.close(fig)
    return (i, j)


def k_sweep_plot(sweep: dict, path: Path, k: int):
    ks = np.array(sweep["k"])
    m, s = np.array(sweep["lsa_mean"]), np.array(sweep["lsa_std"])
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.fill_between(ks, m - s, m + s, color=U_COLOR, alpha=0.15, linewidth=0)
    ax.plot(ks, m, color=U_COLOR, linewidth=2, marker="o", markersize=4)
    ax.axhline(sweep["raw_mean"], color=INK_FAINT, linewidth=1.5)
    ax.text(ks[-1], sweep["raw_mean"] + 0.03, f"raw TF-IDF, one dimension per word: ARI {sweep['raw_mean']:.2f}",
            ha="right", va="bottom", fontsize=10.5, color=INK_SOFT)
    best = int(np.argmax(m))
    kk = list(ks).index(k) if k in ks else best
    ax.scatter([ks[kk]], [m[kk]], s=70, color=S_COLOR, zorder=4)
    ax.annotate(f"k = {ks[kk]}: ARI {m[kk]:.2f}", (ks[kk], m[kk]), xytext=(18, 8), textcoords="offset points",
                fontsize=10.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50, 100, 200], ["1", "2", "3", "5", "10", "20", "50", "100", "200"])
    ax.set_xlabel("number of concepts kept, k (log scale)")
    ax.set_ylabel("ARI vs. true sections")
    ax.set_ylim(0, max(0.7, m.max() + 0.1))
    ax.grid()
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def confusion_pair(ex: Experiment, comp: cl.Comparison, path: Path):
    cats = list(ex.corpus.categories)
    y = ex.labels
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={"wspace": 0.35})
    for ax, lab, t in [
        (axes[0], comp.labels_raw, f"Raw TF-IDF ({ex.tdm.n_terms} dims)   ARI {comp.raw_ari.mean():.2f}"),
        (axes[1], comp.labels_lsa, f"LSA concept space ({comp.k} dims)   ARI {comp.lsa_ari.mean():.2f}"),
    ]:
        names = cl.match_clusters(y, lab, cats)
        M, rows, cols = cl.confusion(y, lab, cats, names)
        _heat(ax, M, rows, [f"C{c + 1}" for c in range(M.shape[1])], SEQ, vmin=0, vmax=100, fontsize=10.5)
        ax.set_title(t, pad=12, fontsize=12, y=1.12)
        ax.set_ylabel("true section")
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("found cluster", labelpad=8)
    fig.savefig(path)
    plt.close(fig)


def similarity_two_ways(ex: Experiment, path: Path, k: int):
    lsa = ex.lsa
    y = ex.labels
    cats = list(ex.corpus.categories)
    order = np.argsort([cats.index(v) for v in y], kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), gridspec_kw={"wspace": 0.08})
    for ax, M, t in [(axes[0], lsa.doc_similarity(), "Cosine similarity using words"),
                     (axes[1], lsa.doc_similarity(k), f"Cosine similarity using {k} concepts")]:
        ax.imshow(np.clip(M[np.ix_(order, order)], 0, 1), cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(t, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        start = 0
        for cat in cats:
            nn = int(np.sum(y == cat))
            ax.text(-8, start + nn / 2, cat, ha="right", va="center", fontsize=10.5, color=INK_SOFT)
            start += nn
    axes[1].set_yticks([])
    for t in axes[1].texts:
        t.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def spectra_compare(exps: dict, path: Path):
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    styles = {"classic": (V_COLOR, "o"), "science": (INK_FAINT, "s"), "news": (U_COLOR, "D")}
    for key, ex in exps.items():
        s = ex.lsa.s[:24]
        col, m = styles[key]
        ax.plot(np.arange(1, len(s) + 1), s / s[0], color=col, marker=m, markersize=4, linewidth=1.8)
        ax.text(len(s) + 0.4, (s / s[0])[-1], {"classic": "classic (9 titles)", "science": "science (24 blurbs)", "news": "news (400 articles)"}[key],
                color=col, fontsize=10, va="center")
    ax.set_xlabel("concept i")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, 31)
    ax.grid()
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def make_all(out_dir: Path, exps: dict, news_k: int, sweep: dict, comp: cl.Comparison) -> dict:
    _setup()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {name: out_dir / f"{name}.png" for name in [
        "classic_matrix", "classic_layers", "classic_synonymy", "scree", "eckart_young",
        "concepts", "doc_map", "k_sweep", "confusion", "spectra", "similarity_two_ways",
    ]}
    classic, news = exps["classic"], exps["news"]
    classic_matrix(classic, p["classic_matrix"])
    classic_layers(classic, p["classic_layers"])
    classic_synonymy(classic, p["classic_synonymy"])
    scree(news, p["scree"], news_k)
    eckart_young(news, p["eckart_young"], news_k)
    concepts(news, p["concepts"])
    axes = doc_map(news, p["doc_map"], news_k)
    k_sweep_plot(sweep, p["k_sweep"], news_k)
    confusion_pair(news, comp, p["confusion"])
    spectra_compare(exps, p["spectra"])
    similarity_two_ways(news, p["similarity_two_ways"], news_k)
    return {"paths": p, "doc_map_axes": axes}
