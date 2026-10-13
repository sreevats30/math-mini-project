import numpy as np
import pandas as pd
import streamlit as st

from app_pages import shared as ui

c = ui.ctx()
tdm, corpus, lsa = c.tdm, c.corpus, c.lsa
m, n = tdm.shape
r = len(lsa.s)

ui.chapter_header(
    3,
    "Factorise A = UΣVᵀ",
    "Every real matrix, of any shape, can be written as a product of three simple matrices: "
    "a rotation, a stretch, and another rotation. That is the singular value decomposition, and it is the "
    "whole engine of LSA.",
)

st.latex(r"A = U\,\Sigma\,V^{\top} = \sum_{i=1}^{r} \sigma_i\, u_i\, v_i^{\top}, \qquad r = \min(m, n) = " + str(r))
ui.svd_diagram(m, n, r, truncated=False)

a, b, cc = st.columns(3, gap="medium")
with a:
    st.markdown(f'<span style="color:{ui.U_COLOR};font-weight:600">U ({m:,} × {r})</span>', unsafe_allow_html=True)
    st.markdown("Columns u₁, u₂, … are orthonormal directions in **word space**. "
                "Each one is a *concept*: a weighted mix of words. They are the eigenvectors of AAᵀ, "
                "which measures which words occur together.")
with b:
    st.markdown(f'<span style="color:{ui.S_COLOR};font-weight:600">Σ ({r} × {r}, diagonal)</span>',
                unsafe_allow_html=True)
    st.markdown("The singular values σ₁ ≥ σ₂ ≥ … ≥ 0. σᵢ says how much of A lies along concept i. "
                "σᵢ² are the eigenvalues of AᵀA (and of AAᵀ).")
with cc:
    st.markdown(f'<span style="color:{ui.V_COLOR};font-weight:600">Vᵀ ({r} × {n:,})</span>',
                unsafe_allow_html=True)
    st.markdown("Rows v₁ᵀ, v₂ᵀ, … are orthonormal directions in **document space**. Column j of Vᵀ "
                "says how strongly document j expresses each concept. They are the eigenvectors of AᵀA.")

ui.key_idea("U and V are the two sides of the same concepts: U says which words make a concept, "
            "V says which documents use it, and σ says how important it is.")

# ----------------------------------------------------------------- numerical checks
st.subheader("Check it numerically", anchor=False)
st.markdown('<div class="lab-prose">The SVD here is computed with <code>numpy.linalg.svd</code>. '
            "Every property the theory promises can be checked on the real matrix. Values around 10⁻¹⁵ are "
            "zero up to floating-point rounding.</div>", unsafe_allow_html=True)
chk = lsa.checks()
v1 = lsa.Vt[0]
eig_res = float(np.linalg.norm(lsa.A.T @ (lsa.A @ v1) - lsa.s[0] ** 2 * v1))
rows = [
    ("A is rebuilt exactly", "‖A − UΣVᵀ‖_F", chk["A_minus_USVt"]),
    ("Columns of U are orthonormal", "‖UᵀU − I‖_F", chk["UtU_minus_I"]),
    ("Columns of V are orthonormal", "‖VᵀV − I‖_F", chk["VtV_minus_I"]),
    ("σᵢ² are the eigenvalues of AᵀA", "max |λᵢ(AᵀA) − σᵢ²|", chk["eig_AtA_vs_sigma2"]),
    ("v₁ is an eigenvector of AᵀA", "‖AᵀA v₁ − σ₁² v₁‖", eig_res),
]
df = pd.DataFrame(
    [{"Property": p, "Measured": q, "Value": f"{v:.1e}", "Holds": "yes" if v < 1e-8 else "no"} for p, q, v in rows]
)
st.dataframe(df, hide_index=True, width="stretch")
st.caption(f"Also: the singular values are sorted ({'yes' if chk['sorted'] else 'no'}) and non-negative "
           f"({'yes' if chk['nonnegative'] else 'no'}). The rank of A is {lsa.rank}.")

with st.expander("Why are U and V eigenvectors? (the two-line proof)", icon=":material/school:"):
    st.latex(r"A^{\top}A = (U\Sigma V^{\top})^{\top}(U\Sigma V^{\top}) = V\Sigma U^{\top}U\Sigma V^{\top} = V\Sigma^{2}V^{\top}")
    st.latex(r"\Rightarrow\; A^{\top}A\,v_i = \sigma_i^{2}\,v_i, \qquad\text{and in the same way}\quad AA^{\top}u_i = \sigma_i^{2}u_i")
    st.markdown("So the SVD is two eigen-decompositions at once: one of the document–document matrix AᵀA "
                "and one of the word–word matrix AAᵀ. Because both are symmetric and positive semi-definite, "
                "their eigenvectors are orthonormal and their eigenvalues are ≥ 0, which is why σᵢ = √λᵢ is real.")

# ----------------------------------------------------------------- layers
st.subheader("A is a sum of rank-1 layers", anchor=False)
st.markdown('<div class="lab-prose">Each term σᵢuᵢvᵢᵀ is a matrix of rank 1: every column is a multiple of '
            "the same word pattern uᵢ. Adding layers one at a time rebuilds A, strongest first.</div>",
            unsafe_allow_html=True)
ui.suggest_classic("The layers are easiest to see on the 12 × 9 example, where every entry is printed.")
max_layers = int(min(lsa.rank, 12))
layers = st.slider("Layers added", 1, max_layers, value=min(2, max_layers), key=f"c3_layers_{c.key}")

docs = ui.sample_docs(c)
terms = ui.top_terms_for(c, docs)
Ar = lsa.reconstruct(layers)
L = lsa.layer(layers - 1)
sl = np.ix_(terms, docs)
lim = float(np.max(np.abs(tdm.A[sl]))) or 1.0
small = len(docs) <= 12
cols = ui.doc_short(c, docs)
rows_lbl = tdm.terms[terms]
ey = lsa.eckart_young(layers)

g1, g2, g3 = st.columns(3)
counts_mode = tdm.weighting == "count"
for col, M, t in [
    (g1, tdm.A[sl], "A"),
    (g2, L[sl], f"layer {layers}: σ{ui.sub(layers)}u{ui.sub(layers)}v{ui.sub(layers)}ᵀ"),
    (g3, Ar[sl], f"sum of the first {layers}"),
]:
    with col:
        fmt = ".0f" if (counts_mode and t == "A") else (".1f" if small else ".2f")
        fig = ui.heatmap(M, rows_lbl, cols, signed=True, text=small, fmt=fmt, zmax=lim,
                         col_names=[ui.doc_label(c, j) for j in docs], row_names=[f"word: {w}" for w in rows_lbl])
        ui.style_fig(fig, height=max(300, 22 * len(terms) + 110), legend=False, title=t)
        fig.update_traces(showscale=False)
        fig.update_xaxes(tickangle=-90 if not small else 0, tickfont=dict(size=10))
        ui.show(fig, key=f"c3_{t}")

ui.measures([
    (f"{ey['frobenius_error']:.3f}", f"‖A − A{ui.sub(layers)}‖, what {layers} layer{'s' if layers > 1 else ''} still miss"),
    (f"{100 * ey['relative_error']:.1f}%", "of the size of A still missing"),
    (f"{100 * ey['energy_kept']:.1f}%", "of the energy ‖A‖² captured"),
])
ui.try_this("Move the slider to 1. The first layer is roughly 'the average document': every column has the "
            "same shape. The second layer is the first *contrast*, and on the 9-title example it already "
            "separates computer-interaction titles from graph-theory titles.")

ui.check_yourself(
    "c3",
    f"U is {m:,} × {r} with orthonormal columns, so UᵀU = I. Is UUᵀ also the identity?",
    ["Yes, always.", "Only when U is square (r = m). Otherwise UUᵀ is a projection onto the span of the uᵢ.",
     "Never."],
    1,
    "UUᵀ is m × m but has rank r. When r < m it projects any word vector onto the r directions in U and "
    "loses everything else. That projection is exactly what folding-in uses in chapter 7.",
)

ui.chapter_footer(3)
