import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app_pages import shared as ui
from lsa_lab import clustering as cl

c = ui.ctx()
tdm, corpus, lsa, k = c.tdm, c.corpus, c.lsa, c.k
labelled = corpus.has_labels

ui.chapter_header(
    6,
    "Cluster the ideas",
    "Now every document is a short, dense vector of concept scores. Run k-means on those vectors and the "
    "documents group by topic, even the ones that share no words.",
)

if k < 2:
    ui.warn("With a single concept every unit-length document vector is either +1 or −1, so there is nothing to "
            "cluster. Set k to at least 2 in the sidebar.")
    ui.chapter_footer(6)
    st.stop()

default_clusters = len(corpus.categories) if labelled else min(3, max(2, corpus.n_docs // 4))
n_clusters = st.number_input("Number of clusters to find", 2, min(12, corpus.n_docs - 1), value=default_clusters,
                             key=f"c6_nclusters_{c.key}",
                             help="k-means needs this up front. The default is the number of true topics.")

st.markdown(
    '<div class="lab-prose">Both methods run the same k-means on unit-length vectors (so distance behaves like '
    f"cosine). The only difference is the representation: <b>raw</b> uses each document's column of A "
    f"({tdm.n_terms:,} numbers, mostly zero), <b>LSA</b> uses its row of V<sub>k</sub>Σ<sub>k</sub> "
    f"({k} numbers). k-means depends on its random start, so each score is the mean over 10 runs.</div>",
    unsafe_allow_html=True,
)

raw = ui.raw_eval(*c.cache_args, int(n_clusters))
lat = ui.lsa_eval(*c.cache_args, k, int(n_clusters))

if labelled:
    y = np.array(corpus.labels)
    ra, la = raw["ARI"].mean(), lat["ARI"].mean()
    ratio = (la / ra) if ra > 0.01 else None
    ui.measures([
        (f"{ra:.2f} ± {raw['ARI'].std():.2f}", "ARI, raw TF-IDF (0 = chance, 1 = perfect)"),
        (f"{la:.2f} ± {lat['ARI'].std():.2f}", f"ARI, LSA concept space with k = {k}"),
        (f"{raw['NMI'].mean():.2f} → {lat['NMI'].mean():.2f}", "normalised mutual information, raw → LSA"),
    ])
    if ratio and ratio > 1.5:
        ui.key_idea(f"Same algorithm, same documents, {ratio:.0f}× better agreement with the real topics, "
                    f"just by moving from {tdm.n_terms:,} word dimensions to {k} concept dimensions.")
    elif la >= 0.99 and ra < 0.99:
        ui.key_idea("LSA recovers the true topics perfectly here, while raw TF-IDF mixes some of them up.")
    elif la >= 0.99 and ra >= 0.99:
        ui.key_idea("On a collection this small and clean both representations find the topics. The difference "
                    "shows on real text: switch to the news articles.")
    else:
        ui.key_idea("Try other values of k in the sidebar and watch the scores move. Chapter 4 plots the whole curve.")

# ----------------------------------------------------------------- similarity before / after
if labelled and corpus.n_docs >= 12:
    st.subheader("The same documents, compared two ways", anchor=False)
    y_ = np.array(corpus.labels)
    order_ = np.argsort([list(corpus.categories).index(v) for v in y_], kind="stable")
    s1, s2 = st.columns(2, gap="medium")
    for col, M, t in [(s1, lsa.doc_similarity(), "cosine similarity using words"),
                      (s2, lsa.doc_similarity(k), f"cosine similarity using {k} concepts")]:
        with col:
            Mo = np.clip(M[np.ix_(order_, order_)], 0, 1)
            names_ = [ui.doc_label(c, j) for j in order_]
            fig = ui.heatmap(Mo, list(range(len(order_))), list(range(len(order_))), text=False, zmax=1.0,
                             row_names=names_, col_names=names_, hover="cos", gap=0)
            fig.update_xaxes(showticklabels=False, type="linear")
            fig.update_yaxes(showticklabels=False, type="linear")
            ui.style_fig(fig, height=430, legend=False, title=t)
            fig.update_traces(showscale=False)
            ui.show(fig, key=f"c6_sim_{t}")
    st.caption("Documents are sorted by true topic (" + ", ".join(corpus.categories) + "), so each bright square "
               "on the diagonal is one topic. Same colour scale in both, from 0 (white) to 1 (dark).")

# ----------------------------------------------------------------- the map
st.subheader("The documents in concept space", anchor=False)
D = lsa.unit_doc_coords(k)
color_by_opts = (["True topic", "Cluster found by LSA"] if labelled else ["Cluster found by LSA"])
cc1, cc2, cc3 = st.columns([1.4, 1, 1])
color_by = cc1.segmented_control("Colour by", color_by_opts, default=color_by_opts[0], key=f"c6_color_{c.key}")
color_by = color_by or color_by_opts[0]
if k >= 2:
    bx, by = lsa.best_axes(k, corpus.labels if labelled else None)
    ax = cc2.selectbox("Horizontal axis", range(k), index=bx, format_func=lambda x: f"concept {x + 1}",
                       key=f"c6_ax_{c.key}_{k}")
    ay = cc3.selectbox("Vertical axis", range(k), index=by, format_func=lambda x: f"concept {x + 1}",
                       key=f"c6_ay_{c.key}_{k}")
else:
    ax = ay = 0

labels_lsa = lat["labels"]
names = cl.match_clusters(corpus.labels, labels_lsa, corpus.categories) if labelled else \
    {cl_: f"Cluster {cl_ + 1}" for cl_ in set(int(x) for x in labels_lsa)}
if color_by == "True topic":
    groups = list(corpus.categories)
    g_of = np.array(corpus.labels)
else:
    order = sorted(names, key=lambda q: (list(corpus.categories).index(names[q]) if names[q] in corpus.categories
                                          else 99, q))
    groups = [f"C{q + 1}: {names[q]}" if labelled else names[q] for q in order]
    lut = {q: g for q, g in zip(order, groups)}
    g_of = np.array([lut[int(q)] for q in labels_lsa])
style = ui.category_style(groups)
fig = go.Figure()
hover = [f"{ui.doc_label(c, j)}<br>{corpus.labels[j] if corpus.labels else ''}" for j in range(corpus.n_docs)]
for g in groups:
    m = np.flatnonzero(g_of == g)
    col, sym, _ = style[g]
    fig.add_trace(go.Scatter(
        x=D[m, ax], y=D[m, ay] if k >= 2 else np.zeros(len(m)), mode="markers", name=g,
        marker=dict(color=col, symbol=sym, size=10 if corpus.n_docs < 50 else 8, opacity=0.85,
                    line=dict(color="white", width=1)),
        text=[hover[j] for j in m], hovertemplate="%{text}<extra></extra>",
    ))
if corpus.n_docs <= 30:
    fig.add_trace(go.Scatter(x=D[:, ax], y=D[:, ay], mode="text", text=ui.doc_short(c, list(range(corpus.n_docs))),
                             textposition="top center", textfont=dict(size=11, color=ui.INK_SOFT), hoverinfo="skip",
                             showlegend=False))
fig.add_hline(y=0, line=dict(color=ui.RULE, width=1))
fig.add_vline(x=0, line=dict(color=ui.RULE, width=1))
ui.style_fig(fig, height=560, legend=True,
             title=f"{corpus.n_docs} documents as unit vectors, seen on two of the {k} concept axes")
fig.update_xaxes(title=f"concept {ax + 1}")
fig.update_yaxes(title=f"concept {ay + 1}", scaleanchor="x", scaleratio=1)
ui.show(fig)
st.caption("A 2-D map can only show two of the k axes, so some groups overlap here that k-means can still "
           "separate in all k dimensions. Try other axis pairs.")

# ----------------------------------------------------------------- confusion
if labelled:
    st.subheader("Which topics get confused?", anchor=False)
    g1, g2 = st.columns(2, gap="large")
    for col, lab, title in [(g1, raw["labels"], "Raw TF-IDF"), (g2, labels_lsa, f"LSA, k = {k}")]:
        nm = cl.match_clusters(corpus.labels, lab, corpus.categories)
        M, rows, cols_ = cl.confusion(corpus.labels, lab, corpus.categories, nm)
        with col:
            fig = ui.heatmap(M, rows, [f"C{i + 1}" for i in range(M.shape[1])], fmt=".0f",
                             zmax=float(max(M.max(), 1)), hover="documents",
                             col_names=[f"cluster C{i + 1}, best matches {cn}" for i, cn in enumerate(cols_)],
                             row_names=[f"true topic: {rr}" for rr in rows])
            ui.style_fig(fig, height=80 + 52 * len(rows), legend=False, title=title)
            fig.update_traces(showscale=False)
            ui.show(fig, key=f"c6_conf_{title}")
    st.caption("Rows are the true topics. Each column is a cluster, matched to the topic it overlaps most "
               "(Hungarian algorithm), so a perfect result is a bright diagonal.")

# ----------------------------------------------------------------- why
st.subheader("Why do these documents belong together?", anchor=False)
st.markdown('<div class="lab-prose">The centre of a cluster in concept space can be mapped back to words with '
            "U<sub>k</sub>. Its top words describe what the cluster is about.</div>", unsafe_allow_html=True)
kw = cl.cluster_keywords(lsa, k, labels_lsa, n=8)
order = sorted(kw, key=lambda q: (list(corpus.categories).index(names[q]) if names[q] in corpus.categories
                                  else 99, q))
cols = st.columns(min(4, len(order)))
for idx_, q in enumerate(order):
    members = np.flatnonzero(labels_lsa == q)
    with cols[idx_ % len(cols)]:
        head = f"C{q + 1}" + (f", mostly {names[q]}" if labelled else "")
        st.markdown(f"**{head}**  \n<span class='lab-small'>{len(members)} documents</span>", unsafe_allow_html=True)
        st.markdown(" ".join(f"`{w}`" for w in kw[q]))
        for j in members[:3]:
            st.markdown(f"<div class='lab-small'>• {ui.doc_label(c, j)}</div>", unsafe_allow_html=True)

with st.expander("Why we do not compare silhouette scores across the two spaces", icon=":material/info:"):
    sr, sl = raw["silhouette"], lat["silhouette"]
    st.markdown(
        f"The silhouette score measures how well separated clusters are *in the space they were found in*. "
        f"Here it is {sr if sr is None else round(sr, 3)} for raw TF-IDF and "
        f"{sl if sl is None else round(sl, 3)} for LSA. But a {k}-dimensional space and a "
        f"{tdm.n_terms:,}-dimensional space have very different distance distributions (in high dimensions "
        "almost all points look equally far apart), so the comparison mostly reflects the dimension. "
        "ARI and NMI compare the clusters to the true labels instead, so they are fair to both."
    )

ui.check_yourself(
    "c6",
    "Why are the document vectors scaled to unit length before k-means?",
    ["To make the SVD faster.",
     "So that clusters depend on the direction of a vector (its mix of concepts), not on how long the document is.",
     "Because k-means only works on vectors of length 1."],
    1,
    "For unit vectors ‖x − y‖² = 2 − 2cos θ, so Euclidean k-means on the unit sphere groups documents by cosine "
    "similarity. A long article and a short one on the same topic then end up together.",
)
ui.chapter_footer(6)
