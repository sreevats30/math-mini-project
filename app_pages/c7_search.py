import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_pages import shared as ui

c = ui.ctx()
tdm, corpus, lsa, k = c.tdm, c.corpus, c.lsa, c.k

ui.chapter_header(
    7,
    "Search by meaning",
    "A search query is just a very short document. Instead of recomputing the SVD, we <i>fold it in</i>: "
    "project it onto the concepts we already have and compare it with every document there.",
)

st.latex(r"\hat q = \Sigma_k^{-1} U_k^{\top} q \quad(\text{a new row of } V_k),\qquad "
         r"\operatorname{score}_j = \cos\!\big(\Sigma_k\hat q,\; \Sigma_k v_j\big) = "
         r"\cos\!\big(U_k^{\top}q,\; U_k^{\top}a_j\big)")
st.markdown('<div class="lab-prose">Both the query and every document are mapped by the same matrix '
            "U<sub>k</sub>ᵀ, so the comparison is fair. Folding a document that is already in A back in gives "
            "exactly its own column of V<sub>k</sub> (the test suite checks this).</div>", unsafe_allow_html=True)

examples = list(corpus.example_queries)
qkey = f"c7_query_{c.key}"
if qkey not in st.session_state:
    st.session_state[qkey] = examples[0] if examples else ""


def _use_example():
    pick = st.session_state.get(f"c7_pick_{c.key}")
    if pick:
        st.session_state[qkey] = pick


if examples:
    st.pills("Example queries", examples, key=f"c7_pick_{c.key}", on_change=_use_example)
query = st.text_input("Your query", key=qkey, placeholder="type a few words")

if not query.strip():
    st.info("Type a query to search the documents.")
    ui.chapter_footer(7)
    st.stop()

if k < 2:
    ui.warn("With a single concept every cosine score is +1 or −1, so the ranking is meaningless. "
            "Set k to at least 2 in the sidebar.")
res = c.ex.search(query, k)
known = res["known_terms"]
if not known:
    ui.warn("None of these words are in the vocabulary, so the query vector q is all zeros and every score is 0. "
            "LSA can only place a query using words it has seen in the collection. This is one of its limits: "
            "a brand-new word has no row in U.")
    ui.chapter_footer(7)
    st.stop()

ignored = [w for w in tdm.vectorizer.build_analyzer()(query) if w not in set(known)]
st.markdown(
    f"Words used: {' '.join(f'`{w}`' for w in known)}"
    + (f"  \nIgnored (stop word or not in the vocabulary): {' '.join(f'`{w}`' for w in ignored)}" if ignored else "")
)

with st.expander("See the folding-in numbers", icon=":material/calculate:"):
    qnz = np.flatnonzero(res["q"])
    st.markdown(f"**q** has {len(qnz)} non-zero entries out of {tdm.n_terms:,}: "
                + ", ".join(f"{tdm.terms[i]} = {res['q'][i]:.3f}" for i in qnz[:12]))
    st.markdown("**q̂ = Σ_k⁻¹ U_kᵀ q**, the query's position as a row of V_k:")
    st.dataframe(pd.DataFrame([res["q_hat"]], columns=[f"concept {i + 1}" for i in range(k)]).style.format("{:+.4f}"),
                 hide_index=True, width="stretch")

raw_hits = int(np.sum(res["raw"] > 0))
order = np.argsort(-res["lsa"], kind="stable")[:10]
zero_in_top = [j for j in order if not res["shared"][j]]
ui.measures([
    (f"{raw_hits}", "documents contain at least one query word (all keyword search can find)"),
    (f"{len(zero_in_top)} of 10", "top LSA results share no word with the query"),
])

rows = []
for rank, j in enumerate(order, start=1):
    rows.append({
        "rank": rank,
        "document": ui.doc_label(c, j),
        "topic": corpus.labels[j] if corpus.labels else "",
        "shared words": ", ".join(res["shared"][j]) if res["shared"][j] else "none",
        "keyword score": float(res["raw"][j]),
        "LSA score": float(res["lsa"][j]),
    })
df = pd.DataFrame(rows)
if not corpus.labels:
    df = df.drop(columns=["topic"])


def _row_style(row):
    found = row["shared words"] == "none"
    return ["background-color: #F7EFDD" if found else "" for _ in row]


st.dataframe(
    df.style.apply(_row_style, axis=1).format({"keyword score": "{:.3f}"}),
    hide_index=True, width="stretch",
    column_config={
        "rank": st.column_config.NumberColumn(width="small"),
        "document": st.column_config.TextColumn(width="large"),
        "LSA score": st.column_config.ProgressColumn(format="%.3f", min_value=-1.0, max_value=1.0),
    },
)
st.caption("Highlighted rows were found by meaning alone: they share no word with the query, so keyword search "
           "scores them 0.")

if k >= 2:
    D = lsa.unit_doc_coords(k)
    qc = res["q_coords"]
    qn = qc / (np.linalg.norm(qc) or 1.0)
    bx, by = lsa.best_axes(k, corpus.labels if corpus.has_labels else None)
    fig = go.Figure()
    if corpus.labels:
        style = ui.category_style(corpus.categories)
        y = np.array(corpus.labels)
        for g in corpus.categories:
            m = np.flatnonzero(y == g)
            col, sym, _ = style[g]
            fig.add_trace(go.Scatter(x=D[m, bx], y=D[m, by], mode="markers", name=g,
                                     marker=dict(color=col, symbol=sym, size=8, opacity=0.35),
                                     text=[ui.doc_label(c, j) for j in m], hovertemplate="%{text}<extra></extra>"))
    else:
        fig.add_trace(go.Scatter(x=D[:, bx], y=D[:, by], mode="markers", name="documents",
                                 marker=dict(color=ui.INK_FAINT, size=8, opacity=0.5),
                                 text=[ui.doc_label(c, j) for j in range(corpus.n_docs)],
                                 hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=D[order, bx], y=D[order, by], mode="markers", name="top 10 results",
                             marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color=ui.INK, width=2)),
                             text=[ui.doc_label(c, j) for j in order], hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[qn[bx]], y=[qn[by]], mode="markers+text", name="your query",
                             marker=dict(size=20, symbol="star", color=ui.S_COLOR, line=dict(color="white", width=1)),
                             text=["query"], textposition="top center", textfont=dict(color=ui.INK),
                             hovertemplate=f"query: {query}<extra></extra>"))
    ui.style_fig(fig, height=520, legend=True, title="Where the query lands in concept space")
    fig.update_xaxes(title=f"concept {bx + 1}")
    fig.update_yaxes(title=f"concept {by + 1}", scaleanchor="x", scaleratio=1)
    ui.show(fig)

ui.try_this("Search the news articles for *computer virus*. Keyword search only finds documents with "
            "'computer' in them, but LSA also ranks Microsoft and software stories that never use either word. "
            "Then raise k in the sidebar to 50: results get more literal, because finer concepts behave more like words.")

ui.check_yourself(
    "c7",
    "Why use folding-in instead of adding the query to A and recomputing the SVD?",
    ["Recomputing would give a different answer every time.",
     "It is cheap: one matrix–vector product with U_k, instead of a new decomposition of the whole collection.",
     "The SVD cannot handle one-word documents."],
    1,
    "Folding-in costs O(mk) per query. A full SVD costs about O(mn·min(m, n)). The price is that the concepts are "
    "frozen: new vocabulary or new topics are not learned until the SVD is recomputed.",
)
ui.chapter_footer(7)
