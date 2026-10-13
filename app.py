"""LSA Lab: an interactive, step-by-step lesson on Latent Semantic Analysis.

Run:  streamlit run app.py      (or: python main.py app)
"""

import streamlit as st

from app_pages import shared

st.set_page_config(
    page_title="LSA Lab: clustering ideas with SVD",
    page_icon=":material/grid_view:",
    layout="wide",
    initial_sidebar_state="expanded",
)

shared.init_state()
shared.css()

pages = {
    "Start here": [
        st.Page("app_pages/home.py", title="The big idea", icon=":material/lightbulb:", default=True),
    ],
    "The pipeline, step by step": [
        st.Page(path, title=f"{i}. {nav}", url_path=path.split("/")[-1][:-3])
        for i, ((path, _), nav) in enumerate(zip(shared.CHAPTERS, shared.NAV_TITLES), start=1)
    ],
    "Reference": [
        st.Page("app_pages/own_text.py", title="Try your own documents", icon=":material/edit_note:"),
        st.Page("app_pages/formulas.py", title="Formula sheet and viva questions", icon=":material/functions:"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("#### Experiment setup")
    st.selectbox(
        "Documents",
        shared.corpus_options(),
        format_func=lambda k: shared.CORPUS_LABELS[k],
        key="corpus_key",
        on_change=shared.on_corpus_change,
    )
    st.segmented_control(
        "Word weighting",
        list(shared.WEIGHTING_LABELS),
        format_func=lambda k: shared.WEIGHTING_LABELS[k],
        key="weighting",
        on_change=shared.keep_weighting,
        help="TF-IDF down-weights words that appear everywhere. The 1990 example uses raw counts.",
    )
    ex = shared.get_experiment(
        st.session_state.corpus_key,
        st.session_state.weighting,
        st.session_state.custom_text if st.session_state.corpus_key == "custom" else "",
    )
    kmax = shared.max_k(ex)
    if kmax > 1:
        # The slider's key includes the collection, so a new collection starts from its own default k.
        kkey = f"k_slider_{st.session_state.corpus_key}_{st.session_state.weighting}"
        if kkey not in st.session_state:
            st.session_state[kkey] = int(min(max(1, st.session_state.k), kmax))
        st.slider(
            "Concepts kept, k",
            1,
            kmax,
            key=kkey,
            help="How many singular values (concepts) to keep. Used from chapter 4 onwards.",
        )
        st.session_state.k = int(st.session_state[kkey])
    else:
        st.session_state.k = 1
        st.caption("Only one concept is available for this corpus.")
    st.caption(f"A is {ex.tdm.n_terms:,} terms × {ex.tdm.n_docs:,} documents.")

nav.run()
