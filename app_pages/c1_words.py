import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_pages import shared as ui

c = ui.ctx()
tdm, corpus = c.tdm, c.corpus

ui.chapter_header(
    1,
    "Documents become word counts",
    "A computer cannot read, but it can count. The first step throws away word order and grammar and "
    "keeps a tally of the meaningful words in each document: its <i>bag of words</i>.",
)
ui.suggest_classic("With 400 articles the counts are hard to take in. The 9-title example shows every word.")

st.subheader("Pick a document", anchor=False)
j = st.selectbox(
    "Document",
    range(corpus.n_docs),
    format_func=lambda d: ui.doc_label(c, d),
    key=f"c1_doc_{c.key}",
    label_visibility="collapsed",
)
kept_tokens = tdm.tokens(corpus.texts[j])
left, right = st.columns([1.5, 1], gap="large")
with left:
    meta = corpus.labels[j] if corpus.labels else ""
    ui.doc_card(corpus.titles[j], meta, ui.highlight_tokens(corpus.texts[j], set(kept_tokens)))
    words_total = len([w for w in corpus.texts[j].split() if w.strip()])
    st.markdown(
        f'<div class="lab-small" style="margin-top:0.6rem"><span class="tok-kept">highlighted</span> words are kept '
        f"({len(kept_tokens)} of about {words_total}). <span class=\"tok-stop\">Grey</span> words are dropped: "
        + ("they are not one of the paper's 12 index terms." if corpus.vocabulary else
           "common English stop words, or words that occur in too few documents to be useful.")
        + "</div>",
        unsafe_allow_html=True,
    )
with right:
    counts = tdm.counts[:, j]
    nz = np.flatnonzero(counts)
    df_bag = pd.DataFrame({"word": tdm.terms[nz], "count": counts[nz].astype(int)}).sort_values(
        ["count", "word"], ascending=[False, True]
    )
    st.markdown(f"**Its bag of words**  \n{len(nz)} distinct words out of a vocabulary of {tdm.n_terms:,}")
    st.dataframe(df_bag, hide_index=True, width="stretch", height=min(360, 38 + 35 * len(df_bag)),
                 column_config={"count": st.column_config.ProgressColumn("count", format="%d", min_value=0,
                                                                         max_value=int(max(2, counts.max())))})

ui.key_idea(
    f"Every document becomes a column of {tdm.n_terms:,} numbers, one per vocabulary word. "
    f"This one has {len(nz)} non-zero entries, so {100 * (1 - len(nz) / tdm.n_terms):.1f}% of its column is zero."
)

st.subheader("The whole collection", anchor=False)
lengths = (tdm.counts > 0).sum(axis=0)
ui.measures([
    (f"{corpus.n_docs:,}", "documents (columns of A)"),
    (f"{tdm.n_terms:,}", "vocabulary words (rows of A)"),
    (f"{np.median(lengths):.0f}", "distinct kept words in a typical document"),
    (f"{100 * tdm.sparsity:.1f}%", "of all entries are zero"),
])

order = np.argsort(-tdm.df)[:15][::-1]
fig = go.Figure(go.Bar(
    x=tdm.df[order], y=tdm.terms[order], orientation="h", marker_color=ui.A_COLOR,
    hovertemplate="%{y}: in %{x} documents<extra></extra>",
))
ui.style_fig(fig, height=430, legend=False, title="Words found in the most documents")
fig.update_xaxes(title="number of documents containing the word")
ui.show(fig)

with st.expander("How the text is cleaned", icon=":material/tune:"):
    st.markdown(
        """
1. **Lower-case** everything, so *Market* and *market* are the same word.
2. **Split into words** of two or more letters. Numbers and punctuation are dropped.
3. **Remove stop words**: very common words such as *the, of, and, is* that appear in almost every document and say nothing about its topic.
4. **Drop rare words** (news corpus only): a word must appear in at least 2 documents. A word seen once cannot connect two documents, so it cannot help find shared topics.

No stemming is done, so *airline* and *airlines* are different words. Chapter 5 shows that LSA learns they are related anyway.
"""
    )

ui.check_yourself(
    "c1",
    "Why are words like 'the' and 'of' removed before building the matrix?",
    [
        "They make the matrix too large for the SVD.",
        "They appear in almost every document, so they make unrelated documents look similar.",
        "They are spelled the same in every language.",
    ],
    1,
    "Words that occur everywhere carry no information about topic, but they would add the same "
    "large component to every document vector and inflate every similarity.",
)

ui.chapter_footer(1)
