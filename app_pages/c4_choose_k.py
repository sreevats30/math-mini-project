import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app_pages import shared as ui

c = ui.ctx()
tdm, corpus, lsa, k = c.tdm, c.corpus, c.lsa, c.k
s = lsa.s
energy = lsa.energy()
cum = np.cumsum(energy)

ui.chapter_header(
    4,
    "Keep the strongest k concepts",
    "Throwing away all but the first k layers gives A<sub>k</sub> = U<sub>k</sub>Σ<sub>k</sub>V<sub>k</sub>ᵀ. "
    "Mathematically it is the best rank-k approximation of A. For text it does something more useful: it removes "
    "the noise of each author's particular choice of words.",
)
st.markdown(f'<div class="lab-small">The number of concepts k is set in the sidebar. It is <b>k = {k}</b> now.</div>',
            unsafe_allow_html=True)

# ----------------------------------------------------------------- spectrum
n_show = int(min(len(s), 60))
idx = np.arange(1, n_show + 1)
left, right = st.columns([1.25, 1], gap="large")
with left:
    fig = go.Figure(go.Bar(
        x=idx, y=s[:n_show],
        marker_color=[ui.S_COLOR if i <= k else "#C9D1DA" for i in idx],
        customdata=np.c_[energy[:n_show] * 100, cum[:n_show] * 100],
        hovertemplate="σ<sub>%{x}</sub> = %{y:.3f}<br>this concept: %{customdata[0]:.2f}% of the energy"
                      "<br>first %{x}: %{customdata[1]:.1f}%<extra></extra>",
    ))
    ui.style_fig(fig, height=360, legend=False,
                 title=f"Singular values, the strength of each concept{' (first 60)' if len(s) > 60 else ''}")
    fig.update_xaxes(title="concept i")
    fig.update_yaxes(title="σᵢ")
    ui.show(fig)
with right:
    fig = go.Figure(go.Scatter(x=np.arange(1, len(s) + 1), y=cum * 100, mode="lines",
                               line=dict(color=ui.S_COLOR, width=2.5),
                               hovertemplate="k = %{x}: %{y:.1f}% kept<extra></extra>"))
    fig.add_trace(go.Scatter(x=[k], y=[cum[k - 1] * 100], mode="markers+text", marker=dict(size=11, color=ui.S_COLOR),
                             text=[f"  k = {k}: {cum[k - 1] * 100:.1f}%"], textposition="middle right",
                             textfont=dict(color=ui.INK), hoverinfo="skip"))
    ui.style_fig(fig, height=360, legend=False, title="Energy kept: share of ‖A‖² in the first k")
    fig.update_xaxes(title="number of concepts kept, k")
    fig.update_yaxes(title="% of the energy", range=[0, 103])
    ui.show(fig)

# ----------------------------------------------------------------- Eckart-Young
st.subheader("Why the SVD truncation is the best you can do", anchor=False)
st.markdown('<div class="lab-prose"><b>Eckart–Young–Mirsky theorem (1936).</b> Among all matrices B of rank at '
            "most k, the truncated SVD is the closest to A, in both the Frobenius and the spectral norm, and "
            "the error is exactly what was thrown away:</div>", unsafe_allow_html=True)
st.latex(r"\min_{\operatorname{rank}B\le k}\lVert A-B\rVert_F=\lVert A-A_k\rVert_F=\sqrt{\sigma_{k+1}^2+\cdots+\sigma_r^2},"
         r"\qquad \lVert A-A_k\rVert_2=\sigma_{k+1}")
ey = lsa.eckart_young(k)
rand = lsa.random_subspace_error(k, trials=20, seed=0)
ui.measures([
    (f"{ey['frobenius_error']:.4f}", f"measured Frobenius error ‖A − A{ui.sub(k)}‖"),
    (f"{ey['frobenius_theory']:.4f}", f"√(σ{ui.sub(k + 1)}² + … + σᵣ²) from the theorem"),
    (f"{ey['spectral_error']:.4f}", f"spectral error ‖A − A{ui.sub(k)}‖₂, equal to σ{ui.sub(k + 1)} = {ey['spectral_theory']:.4f}"),
    (f"{rand.min():.4f}", f"best of 20 random rank-{k} projections: always worse"),
])

ui.key_idea(
    f"With k = {k} the approximation keeps only {100 * ey['energy_kept']:.1f}% of the energy and misses "
    f"{100 * ey['relative_error']:.0f}% of the size of A (its Frobenius norm). That sounds terrible, but the goal is not to rebuild A. "
    "What is thrown away is mostly the particular wording of individual documents; what is kept is the "
    "vocabulary that many documents share."
)

# ----------------------------------------------------------------- usefulness vs k
st.subheader("So how many concepts should we keep?", anchor=False)
if corpus.has_labels and lsa.max_k >= 3:
    n_clusters = len(corpus.categories)
    sweep = ui.k_sweep(*c.cache_args, n_clusters)
    ks = np.array(sweep["k"])
    mu, sd = np.array(sweep["lsa_mean"]), np.array(sweep["lsa_std"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[ks, ks[::-1]], y=np.r_[mu + sd, (mu - sd)[::-1]], fill="toself",
                             fillcolor="rgba(15,122,128,0.15)", line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=ks, y=mu, mode="lines+markers", line=dict(color=ui.U_COLOR, width=2.5),
                             marker=dict(size=7), name="clusters in concept space",
                             hovertemplate="k = %{x}: ARI %{y:.3f}<extra></extra>"))
    fig.add_hline(y=sweep["raw_mean"], line=dict(color=ui.INK_FAINT, width=2),
                  annotation_text=f"raw TF-IDF, one dimension per word: {sweep['raw_mean']:.2f}",
                  annotation_position="bottom right", annotation_font_color=ui.INK_SOFT)
    if k in set(ks):
        mk = mu[list(ks).index(k)]
        fig.add_trace(go.Scatter(x=[k], y=[mk], mode="markers", marker=dict(size=14, color=ui.S_COLOR),
                                 name=f"your k = {k}", hovertemplate=f"k = {k}: ARI %{{y:.3f}}<extra></extra>"))
    ui.style_fig(fig, height=400, legend=True,
                 title="How well k-means recovers the true topics, for each k")
    fig.update_xaxes(type="log", title="number of concepts kept, k (log scale)", tickvals=[2, 3, 5, 10, 20, 50, 100])
    fig.update_yaxes(title="adjusted Rand index (1 = perfect)", rangemode="tozero")
    fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
    ui.show(fig)
    best = int(ks[int(np.argmax(mu))])
    st.markdown(
        f'<div class="lab-prose">Each point is the average of 10 k-means runs. The curve rises while new concepts '
        f"separate real topics, peaks around <b>k = {best}</b>, then falls as extra concepts start to describe "
        "individual documents instead of topics, and the space drifts back towards raw TF-IDF.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="lab-prose">This corpus has no topic labels (or too few concepts), so there is no '
                "ground truth to score against. Use the elbow of the singular values instead.</div>",
                unsafe_allow_html=True)

with st.expander("Choosing k when you have no labels", icon=":material/help:"):
    st.markdown(
        """
- **Elbow of the spectrum.** Keep the concepts before the singular values flatten out. On the news articles the first five or so stand clearly above a long, flat tail.
- **Rule of thumb from the literature.** For large collections LSA usually works best with a few hundred concepts (Deerwester et al. used 100 for thousands of abstracts). Small collections need far fewer.
- **Stability.** Re-run the clustering with different seeds: a good k gives the same clusters every time.
- In this project the labels are only used to *check* the result after the fact, never to build the concept space.
"""
    )

ui.check_yourself(
    "c4",
    "What happens if you keep every concept (k = r)?",
    ["Clustering becomes perfect.", "A_k equals A, so concept space is just a rotated copy of word space.",
     "The SVD fails."],
    1,
    "With k = r nothing is thrown away: ‖A − A_r‖ = 0 and cosine similarities between documents are exactly the "
    "raw TF-IDF ones, because U_r has orthonormal columns and preserves angles. The benefit of LSA comes from "
    "the truncation.",
)
ui.chapter_footer(4)
