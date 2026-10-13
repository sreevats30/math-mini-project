import streamlit as st

from app_pages import shared as ui

c = ui.ctx()
lsa, tdm, k = c.lsa, c.tdm, c.k

st.markdown('<div class="lab-count">LSA Lab: a hands-on lesson in Latent Semantic Analysis</div>', unsafe_allow_html=True)
st.title("Clustering ideas, not words", anchor=False)
st.markdown(
    '<div class="lab-lede">Two documents can be about the same thing without sharing a single word. '
    "Comparing their words cannot see that. Latent Semantic Analysis can, using one matrix factorisation: "
    "the singular value decomposition.</div>",
    unsafe_allow_html=True,
)

pair = c.ex.showcase_pair(k)
if pair is not None:
    i, j, raw_cos, lsa_cos = pair
    st.subheader("Here is the problem, and the fix", anchor=False)
    left, right = st.columns(2)
    for col, d in [(left, i), (right, j)]:
        with col:
            kept = set(tdm.tokens(c.corpus.texts[d]))
            label = c.corpus.labels[d] if c.corpus.labels else ""
            meta = f"{c.corpus.ids()[d]}, {label}" if c.corpus.short_ids else label
            ui.doc_card(c.corpus.titles[d], meta, ui.highlight_tokens(c.corpus.texts[d], kept))
    ui.measures([
        ("0", "indexed words the two documents share"),
        (f"{raw_cos:.2f}", "cosine similarity of their word vectors"),
        (f"{lsa_cos:.2f}", f"cosine similarity after keeping k = {k} concepts"),
    ])
    st.markdown(
        '<div class="lab-prose">Highlighted words are the ones the computer keeps after dropping common words '
        "like <i>the</i> and <i>of</i>. The two lists do not overlap, so as word vectors these documents are "
        "perpendicular. In the concept space found by the SVD they point in almost the same direction, because "
        "the <i>other</i> documents connect their vocabularies.</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("Every pair of documents in this corpus shares at least one word, so there is no showcase pair.")

st.subheader("The one equation behind it", anchor=False)
st.markdown(
    '<div class="lab-prose">Put the documents side by side as the columns of a matrix A, one row per word. '
    f"The SVD splits A into three factors, and keeping only the first k = {k} gives the best possible "
    "rank-k summary of it. Each block below is drawn to scale for the documents you have selected.</div>",
    unsafe_allow_html=True,
)
ui.svd_diagram(tdm.n_terms, tdm.n_docs, k)
a, b, cc = st.columns(3)
a.markdown(f'<span style="color:{ui.U_COLOR};font-weight:600">U: words → concepts.</span> '
           "Each column is a concept written as a weighted list of words.", unsafe_allow_html=True)
b.markdown(f'<span style="color:{ui.S_COLOR};font-weight:600">Σ: how strong each concept is.</span> '
           "The singular values σ₁ ≥ σ₂ ≥ … say how much of A each concept explains.", unsafe_allow_html=True)
cc.markdown(f'<span style="color:{ui.V_COLOR};font-weight:600">Vᵀ: documents → concepts.</span> '
            "Each column places one document in concept space.", unsafe_allow_html=True)

st.subheader("Your route through the lab", anchor=False)
st.markdown('<div class="lab-prose">Seven short chapters, each one step of the pipeline. '
            "Every chart is live: change the documents, the weighting or k in the sidebar and the whole "
            "lesson recomputes.</div>", unsafe_allow_html=True)
blurbs = [
    "Strip a document down to the words that carry meaning.",
    "Stack those word counts into A and weight them with TF-IDF.",
    "Factorise A with the SVD and check every property numerically.",
    "Choose how many concepts to keep, and see why fewer can be better.",
    "Look inside a concept: which words pull it each way.",
    "Cluster documents in concept space and score it against the real topics.",
    "Fold a new query into concept space and search by meaning.",
]
for n, ((path, title), blurb) in enumerate(zip(ui.CHAPTERS, blurbs), start=1):
    cols = st.columns([2.2, 4])
    cols[0].page_link(path, label=f"{n}. {title}")
    cols[1].markdown(f'<div class="lab-small" style="padding-top:0.45rem">{blurb}</div>', unsafe_allow_html=True)

ui.try_this(
    "Start with the 9-title example from the original 1990 paper (choose it under *Documents* in the sidebar). "
    "It is small enough that you can read every number. Then switch to the 400 news articles to see LSA on real text."
)
