import streamlit as st

from app_pages import shared as ui
from lsa_lab.corpora import SAMPLE_CUSTOM_TEXT, custom_corpus
from lsa_lab.text import build_matrix

st.markdown('<div class="lab-count">Reference</div>', unsafe_allow_html=True)
st.title("Try your own documents", anchor=False)
st.markdown(
    '<div class="lab-lede">Paste anything: lecture notes, headlines, reviews, chat messages. One document per line. '
    "To score the clusters against the truth, start a line with its topic and a vertical bar, like "
    "<code>music | The band played the encore</code>.</div>",
    unsafe_allow_html=True,
)

if "own_text_area" not in st.session_state:
    st.session_state.own_text_area = st.session_state.get("custom_text") or SAMPLE_CUSTOM_TEXT

st.text_area("Documents, one per line", key="own_text_area", height=320)
draft = custom_corpus(st.session_state.own_text_area)

problem = None
if draft.n_docs < 3:
    problem = "Add at least 3 documents (non-empty lines)."
else:
    try:
        tdm = build_matrix(draft)
        if tdm.n_terms < 2:
            problem = "After removing stop words fewer than 2 distinct words are left. Add more text."
    except ValueError:
        problem = "After removing stop words there are no words left. Add more text."

c1, c2 = st.columns([1, 3], vertical_alignment="center")
if problem:
    c2.error(problem, icon=":material/error:")
else:
    labels = "with topic labels" if draft.has_labels else "without labels"
    c2.markdown(f'<div class="lab-small">{draft.n_docs} documents {labels}, {tdm.n_terms} distinct words after '
                "cleaning.</div>", unsafe_allow_html=True)


def _use():
    st.session_state.custom_text = st.session_state.own_text_area
    st.session_state.corpus_key = "custom"
    ui.on_corpus_change()


c1.button("Analyse these documents", type="primary", disabled=problem is not None, on_click=_use,
          icon=":material/play_arrow:")

if st.session_state.get("corpus_key") == "custom" and st.session_state.get("custom_text"):
    st.success("Your documents are now the active collection. Every chapter uses them.", icon=":material/check:")
    st.page_link("app_pages/c1_words.py", label="Start at chapter 1 with your documents", icon=":material/arrow_forward:")

with st.expander("What makes a good test collection", icon=":material/tips_and_updates:"):
    st.markdown(
        """
- **At least 3 topics with 4 or more documents each.** LSA learns from words that are shared between documents, so it needs some repetition.
- **Some vocabulary overlap inside a topic, little across topics.** Then the first concepts line up with the topics.
- **Try a trap.** Add a line that belongs to one topic but uses none of its usual words, and see whether LSA still places it correctly in chapter 6.
- Your text stays in this browser session. Nothing is saved.
"""
    )
