# LSA Lab: Complete Technical & Mathematical Guide

**Mini-Project: Mathematical Foundations of Data Science (MFDS)**  
**Topic:** Latent Semantic Analysis (LSA) with Singular Value Decomposition (SVD)

---

## 1. Project Overview & The Big Idea

### The Problem: Why Keyword Search & Raw Word Counts Fail
Traditional NLP representations (such as Bag-of-Words or raw TF-IDF) represent documents as sparse vectors across a giant vocabulary (e.g., 1,400+ distinct words). In this representation:
- **Synonymy:** Two articles discussing the exact same topic using different words (e.g., *"human"* vs. *"user"*, or *"physician"* vs. *"doctor"*) have orthogonal vectors: their dot product and cosine similarity are **0.00**.
- **Sparsity & Noise:** Most documents contain fewer than 50 distinct words. The resulting matrix is 98%+ zeros. High-frequency words or document-specific vocabulary create noise that obscures the underlying topic.
- **Polysemy:** Words with multiple meanings (e.g., *"bank"* = river bank vs. financial institution) are conflated into a single dimension.

### The Solution: Latent Semantic Analysis (LSA)
LSA projects both documents and words from a high-dimensional, sparse **word space** into a low-dimensional, dense **concept space** using the **Singular Value Decomposition (SVD)**:

$$\mathbf{A} \approx \mathbf{A}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^\top$$

By truncating to the top $k$ singular values (e.g., $k = 5$):
1. **Noise is filtered out:** The idiosyncratic quirks of individual documents fall into the discarded tail ($\sigma_{k+1}, \dots, \sigma_r$).
2. **Co-occurrence patterns are uncovered:** If Word $X$ and Word $Y$ frequently appear with Word $Z$, LSA aligns them along the same concept axis, even if $X$ and $Y$ never co-occur in the same sentence.
3. **Clustering performance surges:** On 400 real AG News articles, $K$-means clustering achieves an **Adjusted Rand Index (ARI) of 0.54** in $k=5$ concept space, compared to **0.07** in raw TF-IDF space (**an ~8× improvement**), despite keeping less than 5% of the matrix energy!

---

## 2. Codebase Architecture & File Structure

```
math-mini/
├── app.py                      # Main entry point for the Streamlit web application
├── launch_app.bat              # 1-click Windows launcher (runs streamlit run app.py)
├── main.py                     # CLI entry point: run reports, build figures, rebuild slide deck
├── requirements.txt            # Python dependencies (NumPy, SciPy, Scikit-learn, Streamlit, etc.)
│
├── lsa_lab/                    # Core Python package implementing the mathematics
│   ├── text.py                 # Tokenization, vocabulary, manual TF-IDF weighting, L2 normalization
│   ├── model.py                # Full & truncated SVD, Eckart-Young errors, sign determinism, folding-in
│   ├── clustering.py           # Spherical K-Means, multi-seed ARI/NMI benchmarks, k-parameter sweep
│   ├── corpora.py              # Dataset loaders: Deerwester 1990 (9 docs), Science (24 docs), AG News (400 docs)
│   ├── pipeline.py             # End-to-end experiment pipeline object (Corpus -> TDM -> SVD -> Search)
│   ├── report.py               # Runs numerical evaluations and writes results/metrics.json
│   ├── figures.py              # Renders static publication-quality figures
│   ├── deck.py                 # Programmatically builds the PowerPoint presentation (.pptx) with speaker notes
│   └── style.py                # Cohesive color palette (Teal for U, Ochre for Σ, Violet for V)
│
├── app_pages/                  # The 7 interactive chapters in the Streamlit web app
│   ├── shared.py               # Shared session state, UI components, caching, Plotly charting helpers
│   ├── home.py                 # Chapter 0: The Big Idea (showcase pair, SVD drawn to scale)
│   ├── c1_words.py             # Chapter 1: Tokenization, stop words, Bag-of-Words inspect tool
│   ├── c2_matrix.py            # Chapter 2: The term-document matrix A & step-by-step TF-IDF calculation
│   ├── c3_svd.py               # Chapter 3: Numerical verification of SVD properties (U^T U = I, etc.)
│   ├── c4_choose_k.py          # Chapter 4: Singular values, energy retention, Eckart-Young verification
│   ├── c5_concepts.py          # Chapter 5: Interpreting concepts (top positive & negative words per axis)
│   ├── c6_clusters.py          # Chapter 6: Clustering in raw space vs. concept space (ARI, confusion matrices)
│   ├── c7_search.py            # Chapter 7: Semantic search engine & query folding-in live demo
│   ├── own_text.py             # Custom document playground: paste your own sentences and run LSA live
│   └── formulas.py             # Complete formula reference sheet and 15 viva/interview questions
│
├── data/                       # Offline datasets (AG News sample news_400.csv, science_24.json)
├── figures/                    # Generated charts and diagrams
├── presentation/               # Generated PowerPoint presentation (MFDS_LSA_Presentation.pptx)
├── results/metrics.json        # Single source of truth for benchmark numbers
└── tests/test_lsa.py           # 30 rigorous mathematical and unit tests
```

---

## 3. Mathematical Foundations Explained

Here are all the key mathematical concepts used in this project, explained step by step with their intuition, formulas, and practical role.

### 3.1. The Term-Document Matrix ($\mathbf{A}$) and TF-IDF Weighting

Let $m$ be the number of unique words in our vocabulary, and $n$ be the number of documents in our collection. We construct an $m \times n$ matrix $\mathbf{A}$, where each column represents a document, and each row represents a word.

#### Step A: Raw Count Matrix ($\mathbf{C}$)
$c_{ij}$ is the number of times word $i$ appears in document $j$.

#### Step B: Sublinear Term Frequency (TF)
Instead of using raw count $c_{ij}$, we use:
$$\text{tf}_{ij} = 1 + \ln(c_{ij}) \quad (\text{for } c_{ij} > 0, \text{ else } 0)$$
*Intuition:* A word appearing 20 times in a document is important, but not 20 times more important than a word appearing once. Logarithmic scaling dampens the effect of extreme word repetitions.

#### Step C: Smooth Inverse Document Frequency (IDF)
Words that appear in nearly every document (e.g., *"said"*, *"also"*) contain little discriminative information. IDF gives higher weights to words that appear in only a few documents:
$$\text{idf}_i = \ln\left(\frac{1 + n}{1 + \text{df}_i}\right) + 1$$
where $\text{df}_i$ is the document frequency (how many documents contain word $i$). The $+1$ terms prevent division-by-zero and negative weights.

#### Step D: Unnormalized Weight
$$w_{ij} = \text{tf}_{ij} \times \text{idf}_i = \big(1 + \ln c_{ij}\big)\left(\ln\frac{1 + n}{1 + \text{df}_i} + 1\right)$$

#### Step E: $L_2$ Column Normalization (Cosine Invariance)
To prevent long documents from dominating short ones, every column (document) is scaled to have Euclidean length 1:
$$A_{ij} = \frac{w_{ij}}{\|\mathbf{w}_{\cdot j}\|_2} = \frac{w_{ij}}{\sqrt{\sum_{k=1}^m w_{kj}^2}}$$
After this step, every document vector $\mathbf{a}_j$ is a unit vector on the surface of an $m$-dimensional sphere ($\|\mathbf{a}_j\|_2 = 1$).

---

### 3.2. Singular Value Decomposition (SVD)

The Fundamental Theorem of Linear Algebra states that **any** real $m \times n$ matrix $\mathbf{A}$ can be factorized into three matrices:

$$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

Where:
- $r = \min(m, n)$ is the rank limit.
- $\mathbf{U}$ is an $m \times r$ matrix of orthonormal columns ($\mathbf{U}^\top \mathbf{U} = \mathbf{I}$). Each column $\mathbf{u}_i$ is a **left singular vector** representing **Concept $i$ as a weighted combination of words**.
- $\mathbf{\Sigma}$ is an $r \times r$ diagonal matrix with non-negative entries sorted in descending order:
  $$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$$
  Each $\sigma_i$ is a **singular value**, quantifying the variance or "energy" accounted for by Concept $i$.
- $\mathbf{V}$ is an $n \times r$ matrix of orthonormal columns ($\mathbf{V}^\top \mathbf{V} = \mathbf{I}$). Each column $\mathbf{v}_i$ is a **right singular vector** representing **Concept $i$ across all documents**. The $j$-th row of $\mathbf{V}$ gives Document $j$'s coordinates across the concepts.

#### Relationship to Eigenvalue Decomposition:
- $\mathbf{A}^\top \mathbf{A}$ is an $n \times n$ symmetric document-correlation matrix:
  $$\mathbf{A}^\top \mathbf{A} \mathbf{v}_i = \sigma_i^2 \mathbf{v}_i$$
  $\implies \mathbf{v}_i$ are the eigenvectors of $\mathbf{A}^\top \mathbf{A}$, with eigenvalues $\lambda_i = \sigma_i^2$.
- $\mathbf{A} \mathbf{A}^\top$ is an $m \times m$ symmetric word-co-occurrence matrix:
  $$\mathbf{A} \mathbf{A}^\top \mathbf{u}_i = \sigma_i^2 \mathbf{u}_i$$
  $\implies \mathbf{u}_i$ are the eigenvectors of $\mathbf{A} \mathbf{A}^\top$, with eigenvalues $\lambda_i = \sigma_i^2$.

---

### 3.3. Low-Rank Truncation & The Eckart–Young–Mirsky Theorem

In LSA, we do not keep all $r$ singular values. We pick a small rank $k \ll r$ (e.g., $k = 5$ or $k = 100$) and truncate:

$$\mathbf{A}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^\top = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

Where:
- $\mathbf{U}_k$ is $m \times k$
- $\mathbf{\Sigma}_k$ is $k \times k$
- $\mathbf{V}_k^\top$ is $k \times n$

#### The Eckart–Young–Mirsky Theorem (1936):
For any rank-$k$ matrix $\mathbf{B}$ ($\operatorname{rank}(\mathbf{B}) \le k$), the truncated SVD matrix $\mathbf{A}_k$ is the **optimal approximation** of $\mathbf{A}$ in both Frobenius norm and Spectral norm:

$$\min_{\operatorname{rank}(\mathbf{B}) \le k} \|\mathbf{A} - \mathbf{B}\|_F = \|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$

$$\min_{\operatorname{rank}(\mathbf{B}) \le k} \|\mathbf{A} - \mathbf{B}\|_2 = \|\mathbf{A} - \mathbf{A}_k\|_2 = \sigma_{k+1}$$

*Why this matters:*  
No other linear combination of $k$ basis vectors can reconstruct the matrix with lower squared error. In our code, this theoretical formula is verified live against numerical measurements (`test_eckart_young_exact`).

#### Energy Kept:
$$\text{Energy Ratio} = \frac{\|\mathbf{A}_k\|_F^2}{\|\mathbf{A}\|_F^2} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}$$
*Key Insight:* Even though $k=5$ retains only ~5% of the total matrix energy on text data, clustering accuracy jumps from 0.07 to 0.54! Why? Because the remaining 95% of energy is mostly the noise of unique words occurring in only one document. Filtering it out removes distraction for the clustering algorithm.

---

### 3.4. Concept Coordinates & Semantic Geometry

Once $\mathbf{A}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^\top$ is computed, we can place both documents and words in a shared $k$-dimensional space:

1. **Document Coordinates in Concept Space:**
   $$\hat{\mathbf{d}}_j = \mathbf{U}_k^\top \mathbf{a}_j = \mathbf{\Sigma}_k \mathbf{v}_j \quad \in \mathbb{R}^k$$
   Each document is now represented by $k$ numbers instead of $m$ words.

2. **Word Coordinates in Concept Space:**
   $$\hat{\mathbf{t}}_i = \mathbf{\Sigma}_k \mathbf{u}_i \quad \in \mathbb{R}^k$$
   Each word is represented by $k$ numbers.

#### Solving Synonymy (The Deerwester 1990 Effect):
Because $\mathbf{A}_k \mathbf{A}_k^\top = \mathbf{U}_k \mathbf{\Sigma}_k^2 \mathbf{U}_k^\top$, the similarity between two words in concept space depends on their dot product across all latent concepts.
- In raw space: If *"human"* and *"user"* never appear in the same article title, $\mathbf{a}_{\text{human}}^\top \mathbf{a}_{\text{user}} = 0$ (or correlation = $-0.38$).
- In rank-2 concept space: Because both *"human"* and *"user"* co-occur with words like *"computer"*, *"system"*, and *"interface"*, their latent concept loadings align, causing their correlation to jump to **$+0.94$**!

---

### 3.5. Query Folding-In (Information Retrieval)

How do we search for a new user query $\mathbf{q}$ without having to recompute the entire SVD from scratch?

1. Take the query text and convert it to an $m \times 1$ TF-IDF vector $\mathbf{q}$ (using the exact same vocabulary, IDF weights, and $L_2$ norm as the training documents).
2. **Fold-in transformation:**  
   Since a document satisfies $\mathbf{a}_j \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{v}_j \implies \mathbf{v}_j = \mathbf{\Sigma}_k^{-1} \mathbf{U}_k^\top \mathbf{a}_j$, we project the query as:
   $$\hat{\mathbf{q}} = \mathbf{\Sigma}_k^{-1} \mathbf{U}_k^\top \mathbf{q} \quad \in \mathbb{R}^k$$
3. **Similarity Scoring:**  
   To compare the query with all documents $\mathbf{a}_j$, we compute cosine similarity in the scaled concept space ($\mathbf{\Sigma}_k \hat{\mathbf{q}} = \mathbf{U}_k^\top \mathbf{q}$):
   $$\text{Score}_j = \cos\big(\mathbf{U}_k^\top \mathbf{q},\, \mathbf{\Sigma}_k \mathbf{v}_j\big) = \frac{(\mathbf{U}_k^\top \mathbf{q})^\top (\mathbf{\Sigma}_k \mathbf{v}_j)}{\|\mathbf{U}_k^\top \mathbf{q}\|_2 \|\mathbf{\Sigma}_k \mathbf{v}_j\|_2}$$

*Result:* If you search for *"computer virus"*, LSA returns documents discussing *"security flaws"*, *"trojans"*, and *"system attacks"*, even if those documents never explicitly use the word *"virus"*.

---

### 3.6. Clustering Geometry: Spherical $K$-Means & Cosine Distance

Standard $K$-means uses Euclidean distance $\|\mathbf{x} - \mathbf{y}\|_2^2$. However:
For any two unit vectors $\mathbf{x}$ and $\mathbf{y}$ ($\|\mathbf{x}\| = \|\mathbf{y}\| = 1$):
$$\|\mathbf{x} - \mathbf{y}\|_2^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2 \mathbf{x}^\top \mathbf{y} = 1 + 1 - 2 \cos\theta = 2 - 2 \cos\theta$$

Therefore:
- Minimizing Euclidean distance on unit-normalized vectors is mathematically identical to **maximizing cosine similarity**!
- In `lsa_lab/clustering.py`, document concept vectors are $L_2$-normalized prior to running $K$-means. This is known as **Spherical $K$-Means**.

#### Clustering Evaluation Metrics:
1. **Adjusted Rand Index (ARI):**
   $$\text{ARI} = \frac{\text{Index} - \mathbb{E}[\text{Index}]}{\max(\text{Index}) - \mathbb{E}[\text{Index}]}$$
   Measures how many pairs of documents are correctly assigned to the same cluster vs. different clusters compared to ground truth topic labels. Corrected for chance: a random guess scores **$0.0$**, perfect agreement scores **$1.0$**.
2. **Normalized Mutual Information (NMI):**
   $$\text{NMI}(Y, C) = \frac{2 \cdot I(Y; C)}{H(Y) + H(C)}$$
   Information-theoretic measure of how much information the cluster assignments share with the true class labels. Scales from 0 to 1.

---

### 3.7. Sign Ambiguity in SVD

Notice that in $A = \sum \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$, flipping the signs of both vectors gives:
$$(-\mathbf{u}_i)(-\mathbf{v}_i)^\top = \mathbf{u}_i \mathbf{v}_i^\top$$
Singular vectors are only determined **up to a sign flip** ($\pm 1$). Different BLAS implementations (or running on Intel vs. AMD vs. Mac) might flip the sign of a column of $U$ and $V$.
- **How our codebase solves this:**  
  In `lsa_lab/model.py`, we enforce a canonical sign convention: for each column $\mathbf{u}_i$, we find the word with the largest absolute weight and enforce that its sign is strictly positive. If negative, both $\mathbf{u}_i$ and $\mathbf{v}_i$ are multiplied by $-1$. This ensures 100% reproducible results across all platforms.

---

## 4. How to Run the Project

### Running the Interactive Web Application
You can run the web app with any of these methods:

1. **One-Click Launcher (Windows):**  
   Double click [launch_app.bat](file:///c:/Users/Guest1/Desktop/math-mini/launch_app.bat).

2. **Via Streamlit:**  
   ```powershell
   streamlit run app.py
   ```

3. **Via Python CLI:**  
   ```powershell
   python main.py app
   ```

*(The app will open automatically in your browser at `http://localhost:8501`)*

---

### Running CLI Commands

1. **Run All 30 Math Tests:**  
   ```powershell
   python -m pytest -q
   ```
   *(Verifies SVD orthogonality, Eckart-Young error, folding-in correctness, and clustering determinism)*

2. **Run Analysis & Generate Metrics:**  
   ```powershell
   python main.py report
   ```
   *(Evaluates clustering over 10 random seeds and saves results into `results/metrics.json`)*

3. **Regenerate Publication Figures:**  
   ```powershell
   python main.py figures
   ```
   *(Produces high-resolution figures in `figures/`)*

4. **Rebuild PowerPoint Slide Deck:**  
   ```powershell
   python main.py deck
   ```
   *(Assembles the PowerPoint presentation with graphs, metrics, and presenter notes in `presentation/MFDS_LSA_Presentation.pptx`)*

---

## 5. Viva / Presentation Q&A Cheat Sheet

| Question | Short & Precise Answer |
| :--- | :--- |
| **1. What is LSA in one sentence?** | LSA represents documents in a term-document matrix, applies truncated SVD to find the $k$ most important latent semantic concepts, and compares documents by their concept coordinates rather than literal word counts. |
| **2. Why SVD and not Eigendecomposition?** | The term-document matrix $\mathbf{A}$ is rectangular ($m \times n$, words $\ne$ documents), so it has no eigendecomposition. SVD exists for **any** rectangular matrix. Note that $\mathbf{U}$ and $\mathbf{V}$ are the eigenvectors of the square symmetric matrices $\mathbf{A}\mathbf{A}^\top$ and $\mathbf{A}^\top\mathbf{A}$. |
| **3. What do $\mathbf{U}$, $\mathbf{\Sigma}$, and $\mathbf{V}$ represent?** | $\mathbf{U}$ represents **words in concept space**; $\mathbf{\Sigma}$ represents the **importance/energy of each concept**; $\mathbf{V}$ represents **documents in concept space**. |
| **4. What is the Eckart–Young theorem?** | The truncated SVD $\mathbf{A}_k$ is the mathematically optimal rank-$k$ approximation of $\mathbf{A}$ in both Frobenius and spectral norms. The minimum reconstruction error is exactly $\sqrt{\sum_{i > k} \sigma_i^2}$. |
| **5. If $k=5$ retains under 5% of energy, why does clustering improve 8×?** | Text matrices have a heavy tail of idiosyncratic, document-specific words (noise). The top 5 singular values capture shared vocabulary across documents (the true topics). Discarding 95% of energy eliminates noise, making clustering far more accurate. |
| **6. How do you choose $k$?** | With ground truth labels: Pick $k$ at the peak of the ARI/NMI curve ($k=5$). Without labels: Use the "elbow" of the singular values scree plot, cluster stability across seeds, or domain benchmarks (e.g., $k \approx 100\text{--}300$ for massive corpora). |
| **7. How does query folding-in work?** | Because document columns satisfy $\mathbf{a}_j = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{v}_j \implies \mathbf{v}_j = \mathbf{\Sigma}_k^{-1} \mathbf{U}_k^\top \mathbf{a}_j$, an incoming query $\mathbf{q}$ is projected into the same space using $\hat{\mathbf{q}} = \mathbf{\Sigma}_k^{-1} \mathbf{U}_k^\top \mathbf{q}$, and matched against documents using cosine similarity. |
| **8. Why cosine similarity instead of Euclidean distance?** | Cosine similarity measures angle rather than magnitude. This ensures that a short article and a long article on the exact same topic are recognized as semantically identical regardless of word count. |
| **9. How does LSA handle synonymy and polysemy?** | **Synonymy:** Handled exceptionally well; words sharing similar context vectors align on the same concept axis. **Polysemy:** Handled only partially; LSA assigns a single vector to each word, which becomes an average of its multiple meanings. |
| **10. How is LSA related to PCA?** | PCA is the SVD of the **mean-centered** data matrix. LSA skips mean-centering to preserve matrix sparsity. As a result, the first singular vector in LSA represents the general "average document", while subsequent vectors capture topic variations. |
