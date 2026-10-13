import numpy as np
import streamlit as st

from app_pages import shared as ui

c = ui.ctx()
tdm, corpus, lsa = c.tdm, c.corpus, c.lsa
is_tfidf = tdm.weighting == "tfidf"

ui.chapter_header(
    2,
    "Build the term–document matrix A",
    "Line the bags of words up side by side and you get a matrix: one row per word, one column per document. "
    "This matrix A is the only input the SVD will ever see.",
)
ui.suggest_classic("The 9-title example gives a 12 × 9 matrix you can read in full.")

st.markdown(r"$$A \in \mathbb{R}^{m\times n},\qquad A_{ij} = \text{weight of word } i \text{ in document } j,"
            rf"\qquad m = {tdm.n_terms:,},\; n = {tdm.n_docs:,}$$")

docs = ui.sample_docs(c)
terms = ui.top_terms_for(c, docs)
M = tdm.A[np.ix_(terms, docs)]
title = ("A in full" if len(docs) == tdm.n_docs and len(terms) == tdm.n_terms else
         f"A slice: {len(terms)} of {tdm.n_terms:,} words × {len(docs)} of {tdm.n_docs:,} documents")
fig = ui.heatmap(M, tdm.terms[terms], ui.doc_short(c, docs), fmt=".0f" if not is_tfidf else ".2f",
                 text=len(docs) <= 12 or not is_tfidf, row_names=[f"word: {t}" for t in tdm.terms[terms]],
                 col_names=[ui.doc_label(c, j) for j in docs], hover="A")
ui.style_fig(fig, height=max(300, 24 * len(terms) + 110), legend=False, title=title)
ui.show(fig)
if len(docs) < tdm.n_docs:
    st.caption("Columns: a balanced sample of documents from each topic. Hover a cell to see the full title.")

ui.key_idea(
    f"{100 * tdm.sparsity:.1f}% of the entries of A are zero. Most words are missing from most documents, "
    "which is exactly why comparing word vectors directly works so badly."
)

# ---------------------------------------------------------------- weighting
st.subheader("Weighting: raw counts or TF-IDF?", anchor=False)
st.markdown(
    '<div class="lab-prose">Raw counts treat every word equally. TF-IDF boosts words that are frequent '
    "<i>in this document</i> but rare <i>in the collection</i>, and scales every column to length 1 so "
    "long and short documents are comparable.</div>",
    unsafe_allow_html=True,
)
st.latex(r"\mathrm{tf}_{ij} = 1+\ln(\text{count}_{ij}),\quad "
         r"\mathrm{idf}_i = \ln\frac{1+n}{1+\mathrm{df}_i}+1,\quad "
         r"w_{ij} = \mathrm{tf}_{ij}\cdot\mathrm{idf}_i,\quad "
         r"A_{ij} = \frac{w_{ij}}{\lVert w_{\cdot j}\rVert_2}")

st.markdown("**Work one entry out by hand**")
l, r = st.columns(2)
with l:
    j = st.selectbox("Document", range(corpus.n_docs), format_func=lambda d: ui.doc_label(c, d),
                     index=docs[0], key=f"c2_doc_{c.key}")
present = np.flatnonzero(tdm.counts[:, j] > 0)
with r:
    choices = list(tdm.terms[present]) or list(tdm.terms[:1])
    t = st.selectbox("Word", choices, key=f"c2_term_{c.key}_{j}")
b = tdm.tfidf_breakdown(t, j)
if is_tfidf:
    st.latex(
        rf"\text{{count}} = {b['count']:.0f}\;\Rightarrow\; \mathrm{{tf}} = 1+\ln {b['count']:.0f} = {b['tf']:.3f}"
    )
    st.latex(
        rf"\mathrm{{df}} = {b['df']} \text{{ of }} n = {b['n']} \;\Rightarrow\; "
        rf"\mathrm{{idf}} = \ln\tfrac{{1+{b['n']}}}{{1+{b['df']}}}+1 = {b['idf']:.3f}"
    )
    st.latex(
        rf"w = {b['tf']:.3f}\times{b['idf']:.3f} = {b['w']:.3f},\qquad "
        rf"\lVert w_{{\cdot j}}\rVert = {b['norm']:.3f}\;\Rightarrow\; A_{{ij}} = \tfrac{{{b['w']:.3f}}}{{{b['norm']:.3f}}} = {b['a']:.4f}"
    )
    st.caption(f"Words in fewer documents get a larger idf. '{t}' appears in {b['df']} of {b['n']} documents.")
else:
    st.latex(rf"A_{{ij}} = \text{{count of '{t}' in this document}} = {b['count']:.0f}")
    st.caption("With raw counts no weighting is applied. Switch to TF-IDF in the sidebar to see the full calculation.")

# ---------------------------------------------------------------- why raw similarity fails
st.subheader("Why comparing columns of A is not enough", anchor=False)
st.markdown(
    r"The natural similarity between two documents is the cosine of the angle between their columns, "
    r"$\cos\theta = \dfrac{a_i^\top a_j}{\lVert a_i\rVert\,\lVert a_j\rVert}$. "
    "If two documents share no word their columns are perpendicular and the cosine is exactly 0, "
    "however close their topics are."
)
R = lsa.doc_similarity()
overlap = (tdm.counts > 0).astype(int)
shared = overlap.T @ overlap
n = tdm.n_docs
iu = np.triu_indices(n, 1)
if corpus.labels:
    y = np.array(corpus.labels)
    same = (y[:, None] == y[None, :])[iu]
    zero_same = float(np.mean(shared[iu][same] == 0)) if same.any() else 0.0
    ui.measures([
        (f"{100 * zero_same:.0f}%", "of same-topic document pairs share no word at all"),
        (f"{R[iu][same].mean():.3f}", "average cosine, same topic"),
        (f"{R[iu][~same].mean():.3f}", "average cosine, different topics"),
    ])
    order = np.argsort([corpus.categories.index(v) for v in y], kind="stable")
else:
    zero = float(np.mean(shared[iu] == 0))
    ui.measures([(f"{100 * zero:.0f}%", "of document pairs share no word at all"),
                 (f"{R[iu].mean():.3f}", "average cosine between documents")])
    order = np.arange(n)

show_n = order if n <= 400 else order[:400]
names = [ui.doc_label(c, d) for d in show_n]
zcap = 1.0 if n <= 30 else 0.3
fig = ui.heatmap(R[np.ix_(show_n, show_n)], list(range(len(show_n))), list(range(len(show_n))), text=n <= 12,
                 fmt=".2f", zmax=zcap, row_names=names, col_names=names, hover="cos", gap=0 if n > 40 else 1.5)
fig.update_xaxes(showticklabels=n <= 30, type="linear" if n > 30 else "category",
                 ticktext=ui.doc_short(c, list(show_n)), tickvals=list(range(len(show_n))))
fig.update_yaxes(showticklabels=n <= 30, type="linear" if n > 30 else "category",
                 ticktext=ui.doc_short(c, list(show_n)), tickvals=list(range(len(show_n))))
ui.style_fig(fig, height=520 if n > 30 else 420, legend=False,
             title="Cosine similarity between every pair of documents, in word space")
ui.show(fig)
if corpus.labels:
    st.caption("Documents are sorted by topic, so a good representation would show bright squares along the "
               "diagonal. In word space the squares are faint"
               + (" (the colour scale stops at 0.3 so that faint values are visible)" if zcap < 1 else "")
               + ". Chapter 6 shows the same picture in concept space.")

ui.check_yourself(
    "c2",
    "A word appears in every document of the collection. What is its idf (with the formula above)?",
    ["0", "1", "ln(n)", "It depends on how often it appears."],
    1,
    "df = n, so idf = ln((1+n)/(1+n)) + 1 = ln 1 + 1 = 1. The smoothed formula never fully deletes a word, "
    "but rare words get up to ln((1+n)/2) + 1, several times more weight.",
)

ui.chapter_footer(2)
