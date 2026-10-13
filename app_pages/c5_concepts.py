import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app_pages import shared as ui

c = ui.ctx()
tdm, corpus, lsa, k = c.tdm, c.corpus, c.lsa, c.k

ui.chapter_header(
    5,
    "Read the concepts",
    "A concept is a column uᵢ of U: one weight for every word. Words with large positive weights pull one way, "
    "words with large negative weights pull the other. Reading the extremes tells you what the concept means.",
)

# ----------------------------------------------------------------- one concept
if k > 1:
    i = st.select_slider("Concept", options=list(range(1, k + 1)), value=min(2, k), key=f"c5_concept_{c.key}_{k}",
                         help="Only the k concepts you keep (sidebar) are shown.") - 1
else:
    i = 0
con = lsa.concept(i, n_terms=10, n_docs=3)
items = sorted(con.negative + con.positive, key=lambda x: x[1])
left, right = st.columns([1.15, 1], gap="large")
with left:
    fig = go.Figure(go.Bar(
        x=[w for _, w in items], y=[t for t, _ in items], orientation="h",
        marker_color=[ui.DIVERGING[0] if w < 0 else ui.DIVERGING[-1] for _, w in items],
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    ui.style_fig(fig, height=max(320, 24 * len(items) + 90), legend=False,
                 title=f"Concept {i + 1}: σ{i + 1} = {con.sigma:.3f}, {100 * lsa.energy()[i]:.1f}% of the energy")
    fig.update_xaxes(title=f"weight of the word in u{i + 1}", zeroline=True, zerolinecolor=ui.INK_FAINT)
    fig.update_yaxes(type="category")
    ui.show(fig)
with right:
    st.markdown("**Documents at the positive end**")
    for j, v in con.top_docs:
        lab = f" ({corpus.labels[j]})" if corpus.labels else ""
        st.markdown(f"- {ui.doc_label(c, j)}{lab}  \n  <span class='lab-small'>v = {v:+.3f}</span>",
                    unsafe_allow_html=True)
    if con.negative:
        st.markdown("**Documents at the negative end**")
        for j, v in con.bottom_docs:
            lab = f" ({corpus.labels[j]})" if corpus.labels else ""
            st.markdown(f"- {ui.doc_label(c, j)}{lab}  \n  <span class='lab-small'>v = {v:+.3f}</span>",
                        unsafe_allow_html=True)

if i == 0 and not con.negative:
    ui.key_idea("Concept 1 has no negative words. A has no negative entries, so its strongest direction is roughly "
                "the average document: the words that are common across the whole collection. The topics show "
                "up from concept 2 onwards, as contrasts.")
else:
    ui.key_idea("Concepts after the first are contrasts: a document scores high when it uses the red words and low "
                "when it uses the blue ones. A topic usually appears as one end of one or more concepts.")

with st.expander("Why can the signs flip?", icon=":material/swap_vert:"):
    st.markdown("If (uᵢ, vᵢ) is a pair of singular vectors, so is (−uᵢ, −vᵢ): the product σᵢuᵢvᵢᵀ is unchanged. "
                "Different libraries may return either sign. This lab fixes the sign so that the word with the "
                "largest weight is positive, which makes the output repeatable.")

# ----------------------------------------------------------------- word map
st.subheader("Words in concept space", anchor=False)
st.markdown('<div class="lab-prose">Each word i becomes the point Σ<sub>k</sub>uᵢ (row i of U<sub>k</sub>Σ<sub>k</sub>). '
            "Words used in the same kinds of documents land near each other, whether or not they ever appear "
            "together.</div>", unsafe_allow_html=True)
if k >= 2:
    a1, a2, _ = st.columns([1, 1, 2])
    default = (1, 2) if k >= 3 else (0, 1)
    ax = a1.selectbox("Horizontal axis", range(1, k + 1), index=default[0], key=f"c5_ax_{c.key}_{k}",
                      format_func=lambda x: f"concept {x}") - 1
    ay = a2.selectbox("Vertical axis", range(1, k + 1), index=default[1], key=f"c5_ay_{c.key}_{k}",
                      format_func=lambda x: f"concept {x}") - 1
    T = lsa.term_coords(k)
    n_words = min(80, tdm.n_terms)
    n_labels = min(28, n_words)
    strength = np.linalg.norm(T[:, [ax, ay]], axis=1)
    top = np.argsort(-strength)[:n_words]
    fig = go.Figure(go.Scatter(
        x=T[top, ax], y=T[top, ay], mode="markers", customdata=tdm.terms[top],
        marker=dict(size=7, color=ui.U_COLOR, opacity=0.55),
        hovertemplate="%{customdata}<br>(%{x:.3f}, %{y:.3f})<extra></extra>",
    ))
    lab = top[:n_labels]
    fig.add_trace(go.Scatter(
        x=T[lab, ax], y=T[lab, ay], mode="markers+text", text=tdm.terms[lab], textposition="top center",
        textfont=dict(size=12, color=ui.INK), marker=dict(size=8, color=ui.U_COLOR),
        hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f})<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=ui.RULE, width=1))
    fig.add_vline(x=0, line=dict(color=ui.RULE, width=1))
    ui.style_fig(fig, height=560, legend=False,
                 title=f"The {n_words} words furthest from the origin on these axes ({n_labels} labelled)")
    fig.update_xaxes(title=f"concept {ax + 1}")
    fig.update_yaxes(title=f"concept {ay + 1}")
    ui.show(fig)

# ----------------------------------------------------------------- synonymy
st.subheader("Do two words mean similar things?", anchor=False)
st.markdown('<div class="lab-prose">Compare two rows of the matrix: how the words are spread across the documents. '
            "In A, two words that never appear in the same document can only look unrelated. In A<sub>k</sub> each "
            "word row is rebuilt from the shared concepts, so related words line up.</div>", unsafe_allow_html=True)
vocab = list(tdm.terms)
pairs = [p for p in corpus.word_pairs if p[0] in vocab and p[1] in vocab]
p0 = pairs[0] if pairs else (vocab[0], vocab[min(1, len(vocab) - 1)])
if pairs:
    pick = st.segmented_control("Suggested pairs", [f"{a} / {b}" for a, b in pairs], key=f"c5_pairs_{c.key}",
                                default=f"{p0[0]} / {p0[1]}")
    if pick:
        p0 = tuple(pick.split(" / "))
w1c, w2c, _ = st.columns([1, 1, 2])
w1 = w1c.selectbox("First word", vocab, index=vocab.index(p0[0]), key=f"c5_w1_{c.key}_{p0[0]}")
w2 = w2c.selectbox("Second word", vocab, index=vocab.index(p0[1]), key=f"c5_w2_{c.key}_{p0[1]}")
i1, i2 = vocab.index(w1), vocab.index(w2)
together = int(np.sum((tdm.counts[i1] > 0) & (tdm.counts[i2] > 0)))
r0 = lsa.term_correlation(w1, w2)
rk = lsa.term_correlation(w1, w2, k)
ui.measures([
    (f"{together}", f"documents that contain both '{w1}' and '{w2}'"),
    ("n/a" if r0 is None else f"{r0:+.2f}", "correlation of their rows in A"),
    ("n/a" if rk is None else f"{rk:+.2f}", f"correlation of their rows in A{ui.sub(k)}"),
])
if together == 0 and rk is not None and rk > 0.5:
    st.markdown(f"These two words are never used together, yet after keeping {k} concepts they are strongly "
                "correlated. That is LSA learning that they belong to the same topic, the effect the 1990 paper "
                "called solving the **synonymy** problem.")
if rk is not None and rk > 0.98 and k <= 8:
    st.caption(f"With only {k} concepts every word row in A{ui.sub(k)} is a mix of {k} patterns, so words from the "
               "same topic become almost perfectly correlated. Raise k to see finer differences.")
if corpus.key == "classic":
    st.caption("The 1990 paper reports exactly these numbers for human / user: −0.38 in the original matrix and "
               "0.94 at rank 2. The test suite checks that this code reproduces them.")

if tdm.n_terms <= 15:
    idx = np.arange(tdm.n_terms)
    g1, g2 = st.columns(2)
    for col, M, t in [(g1, lsa.term_correlation_matrix(idx), "Word–word correlation in A"),
                      (g2, lsa.term_correlation_matrix(idx, k), f"… and in A{ui.sub(k)}")]:
        with col:
            fig = ui.heatmap(M, tdm.terms, tdm.terms, signed=True, fmt=".1f", zmax=1.0, hover="r")
            ui.style_fig(fig, height=520, legend=False, title=t)
            fig.update_traces(showscale=False)
            fig.update_xaxes(tickangle=-60)
            fig.update_layout(margin=dict(t=150))
            ui.show(fig, key=f"c5_{t}")

ui.check_yourself(
    "c5",
    "Two words never appear in the same document. How can LSA still make them similar?",
    ["It looks the words up in a dictionary.",
     "They both co-occur with the same other words, so they load on the same concepts.",
     "It cannot; LSA only uses co-occurrence within a document."],
    1,
    "This is second-order co-occurrence. 'human' and 'user' never share a title, but both appear with "
    "'computer', 'interface' and 'system'. The SVD builds concepts from those shared patterns, so both words "
    "get similar coordinates.",
)
ui.chapter_footer(5)
