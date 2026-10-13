import pandas as pd
import streamlit as st

from app_pages import shared as ui

c = ui.ctx()
m, n, k = c.tdm.n_terms, c.tdm.n_docs, c.k

st.markdown('<div class="lab-count">Reference</div>', unsafe_allow_html=True)
st.title("Formula sheet and viva questions", anchor=False)
st.markdown('<div class="lab-lede">Everything from the seven chapters on one page, with the shapes for the '
            "collection you have selected.</div>", unsafe_allow_html=True)

st.subheader("Notation", anchor=False)
st.dataframe(pd.DataFrame([
    ("A", f"{m:,} × {n:,}", "term–document matrix: rows are words, columns are documents"),
    ("aⱼ", f"{m:,} × 1", "column j of A, document j as a word vector"),
    ("U", f"{m:,} × r", "left singular vectors: each column is a concept as a list of word weights, UᵀU = I"),
    ("Σ", "r × r", "diagonal matrix of singular values σ₁ ≥ σ₂ ≥ … ≥ σᵣ ≥ 0"),
    ("V", f"{n:,} × r", "right singular vectors: row j gives document j's weight on each concept, VᵀV = I"),
    ("r", f"{min(m, n):,}", "number of singular values, min(m, n)"),
    ("k", f"{k}", "number of concepts kept"),
    ("A_k", f"{m:,} × {n:,}", "rank-k approximation U_k Σ_k V_kᵀ"),
    ("q", f"{m:,} × 1", "a query, weighted exactly like a document"),
], columns=["symbol", "shape", "meaning"]), hide_index=True, width="stretch")

st.subheader("The pipeline in seven formulas", anchor=False)
st.markdown("**1. Weight the words (TF-IDF)**")
st.latex(r"A_{ij}=\frac{w_{ij}}{\lVert w_{\cdot j}\rVert_2},\quad w_{ij}=\big(1+\ln c_{ij}\big)\Big(\ln\tfrac{1+n}{1+\mathrm{df}_i}+1\Big)")
st.markdown("**2. Factorise (SVD)**")
st.latex(r"A=U\Sigma V^{\top}=\sum_{i=1}^{r}\sigma_i u_i v_i^{\top},\qquad A^{\top}Av_i=\sigma_i^2 v_i,\quad AA^{\top}u_i=\sigma_i^2u_i")
st.markdown("**3. Truncate (Eckart–Young–Mirsky)**")
st.latex(r"A_k=U_k\Sigma_kV_k^{\top},\qquad \lVert A-A_k\rVert_F=\min_{\operatorname{rank}B\le k}\lVert A-B\rVert_F=\sqrt{\textstyle\sum_{i>k}\sigma_i^2}")
st.markdown("**4. Energy kept**")
st.latex(r"\frac{\lVert A_k\rVert_F^2}{\lVert A\rVert_F^2}=\frac{\sum_{i\le k}\sigma_i^2}{\sum_{i\le r}\sigma_i^2}")
st.markdown("**5. Coordinates in concept space**")
st.latex(r"\hat d_j = U_k^{\top}a_j=\Sigma_k v_j\ \ (\text{documents}),\qquad \hat t_i=\Sigma_k u_i\ \ (\text{words})")
st.markdown("**6. Fold in a query**")
st.latex(r"\hat q=\Sigma_k^{-1}U_k^{\top}q,\qquad \mathrm{score}_j=\cos(\Sigma_k\hat q,\ \Sigma_kv_j)")
st.markdown("**7. Compare and cluster**")
st.latex(r"\cos\theta=\frac{x^{\top}y}{\lVert x\rVert\,\lVert y\rVert},\qquad \text{for unit vectors }\ \lVert x-y\rVert^2=2-2\cos\theta")

st.subheader("Questions an examiner might ask", anchor=False)
QA = [
    ("What is LSA in one sentence?",
     "Represent documents as columns of a weighted term–document matrix, take its truncated SVD, and compare documents "
     "by their coordinates on the k strongest singular directions instead of by their raw words."),
    ("Why the SVD and not an eigendecomposition of A?",
     "A is rectangular (words × documents), so it has no eigendecomposition. The SVD exists for every real matrix. "
     "It is closely related: V and U are the eigenvectors of AᵀA and AAᵀ, and σᵢ² their eigenvalues."),
    ("What do U, Σ and V mean in this project?",
     "Each column of U is a concept written as word weights; each σᵢ is how much of A that concept accounts for; each "
     "row of V (scaled by Σ) is a document's position on the concepts."),
    ("State the Eckart–Young theorem and say why it matters here.",
     "The truncated SVD A_k is the closest rank-k matrix to A in Frobenius and spectral norm, with error √(Σ_{i>k} σᵢ²) "
     "and σ_{k+1}. It guarantees that keeping the top k singular triplets is the best possible linear k-dimensional "
     "summary of the collection. The app verifies the error formula numerically."),
    ("If k = 5 keeps under 5% of the energy, how can it be good?",
     "Text matrices have a long flat tail of small singular values, mostly the idiosyncratic words of single documents. "
     "The few strong concepts are the vocabulary many documents share, which is what topic clustering needs. "
     "Reconstruction accuracy and usefulness for clustering are different goals."),
    ("How do you choose k?",
     "With labels, from the clustering-quality curve (chapter 4), which rises, peaks and falls. Without labels, from the "
     "elbow of the singular values, from stability of the clusters across seeds, or a literature default (a few "
     "hundred for large collections)."),
    ("How does folding-in work and why is there a Σ⁻¹?",
     "A document column satisfies aⱼ = U Σ vⱼ, so vⱼ = Σ⁻¹Uᵀaⱼ. Applying the same map to a query gives q̂ = Σ_k⁻¹U_kᵀq, "
     "a new row of V_k. Scores are cosines in the scaled space Σ_k q̂ = U_kᵀq, the same space the documents live in."),
    ("Why cosine similarity and why unit-length vectors for k-means?",
     "Cosine ignores vector length, so a long and a short article on the same topic are similar. For unit vectors "
     "‖x − y‖² = 2 − 2cos θ, so Euclidean k-means on normalised vectors clusters by angle."),
    ("How does LSA handle synonymy and polysemy?",
     "Synonymy well: words used in the same contexts get similar concept vectors even if they never co-occur "
     "(human/user: correlation −0.38 → 0.94). Polysemy only partly: each word gets one vector, an average of its senses."),
    ("What are ARI and NMI?",
     "Ways to compare a clustering with the true labels that do not care about cluster numbering. The adjusted Rand index "
     "counts pairs of documents grouped consistently, corrected so that random labels score 0. Normalised mutual "
     "information measures how much knowing the cluster tells you about the label, from 0 to 1."),
    ("Why not compare silhouette scores between raw and LSA space?",
     "The silhouette is computed with distances inside each space. Distances in a "
     f"{m:,}-dimensional sparse space and a {k}-dimensional dense space have very different distributions, so the "
     "numbers mostly reflect dimension, not quality."),
    ("How is LSA related to PCA?",
     "PCA is the SVD of the mean-centred data matrix. LSA uses the SVD of A without centring, which keeps A sparse. The "
     "first LSA concept is therefore roughly the average document rather than a direction of variation."),
    ("What does it cost to compute?",
     f"A full SVD of an m × n matrix costs about O(mn·min(m, n)). Here ({m:,} × {n:,}) that takes well under a second. "
     "For millions of documents you would compute only the top k triplets with a Lanczos or randomised SVD, roughly O(mnk)."),
    ("What are the main limitations?",
     "Bag of words loses word order and negation; concepts have negative weights and are not probabilities (LDA and NMF "
     "fix this); the Frobenius loss assumes Gaussian noise, which word counts are not; and the vocabulary and concepts "
     "are frozen until the SVD is recomputed."),
    ("Why did the first version of this project show weak results?",
     "Its 24 one-sentence documents barely share words, so A has almost no co-occurrence structure: its singular values "
     "are nearly flat (1.22, 1.17, 1.15 …) and 4 concepts keep only 22% of the energy. LSA needs repeated vocabulary "
     "across documents to find concepts. The 400 news articles have it."),
]
for q, a in QA:
    with st.expander(q):
        st.markdown(a)

st.subheader("Sources", anchor=False)
st.markdown(
    """
- Deerwester, Dumais, Furnas, Landauer and Harshman (1990). *Indexing by Latent Semantic Analysis.* Journal of the American Society for Information Science 41(6), 391–407.
- Landauer, Foltz and Laham (1998). *An Introduction to Latent Semantic Analysis.* Discourse Processes 25, 259–284.
- Eckart and Young (1936). *The approximation of one matrix by another of lower rank.* Psychometrika 1, 211–218.
- Zhang, Zhao and LeCun (2015). *Character-level Convolutional Networks for Text Classification* (source of the AG News split).
"""
)
