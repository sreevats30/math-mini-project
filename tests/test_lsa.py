"""Checks that the linear algebra in lsa_lab is right. Run with:  python -m pytest -q"""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from lsa_lab import clustering as cl
from lsa_lab import custom_corpus, load_corpus, run_experiment
from lsa_lab.text import TOKEN_PATTERN


@pytest.fixture(scope="module", params=["classic", "science", "news"])
def ex(request):
    return run_experiment(request.param)


def test_tfidf_matches_scikit_learn():
    corpus = load_corpus("science")
    ours = run_experiment(corpus).tdm
    ref = TfidfVectorizer(token_pattern=TOKEN_PATTERN, stop_words="english", sublinear_tf=True)
    X = ref.fit_transform(corpus.texts).toarray().T
    assert list(ref.get_feature_names_out()) == list(ours.terms)
    assert np.allclose(ours.A, X)


def test_documents_have_unit_length():
    A = run_experiment("news").tdm.A
    assert np.allclose(np.linalg.norm(A, axis=0), 1.0)


def test_svd_reconstructs_and_is_orthonormal(ex):
    c = ex.lsa.checks()
    assert c["A_minus_USVt"] < 1e-10
    assert c["UtU_minus_I"] < 1e-10
    assert c["VtV_minus_I"] < 1e-10
    assert c["sorted"] and c["nonnegative"]
    assert c["eig_AtA_vs_sigma2"] < 1e-10


def test_sign_convention_keeps_A(ex):
    lsa = ex.lsa
    assert np.allclose((lsa.U * lsa.s) @ lsa.Vt, lsa.A)
    top = np.abs(lsa.U).argmax(axis=0)
    assert np.all(lsa.U[top, np.arange(lsa.U.shape[1])] > 0)


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_eckart_young(ex, k):
    ey = ex.lsa.eckart_young(k)
    assert ey["frobenius_error"] == pytest.approx(ey["frobenius_theory"], rel=1e-9, abs=1e-12)
    assert ey["spectral_error"] == pytest.approx(ey["spectral_theory"], rel=1e-6, abs=1e-12)
    # no random rank-k subspace does better than the SVD
    assert ex.lsa.random_subspace_error(k, trials=5).min() >= ey["frobenius_error"] - 1e-9


def test_rank1_layers_sum_to_A(ex):
    lsa = ex.lsa
    total = sum(lsa.layer(i) for i in range(len(lsa.s)))
    assert np.allclose(total, lsa.A)


def test_folding_in_a_document_recovers_its_coordinates(ex):
    lsa, k = ex.lsa, 2
    j = 0
    a_j = lsa.A[:, j]
    assert np.allclose(lsa.fold_in(a_j, k), lsa.Vt[:k, j])
    assert np.allclose(lsa.query_coords(a_j, k), lsa.doc_coords(k)[j])
    assert lsa.latent_similarity(a_j, k)[j] == pytest.approx(1.0)


def test_classic_synonymy_result_from_the_paper():
    """Deerwester et al. (1990): r(human, user) goes from -0.38 to 0.94 at rank 2."""
    lsa = run_experiment("classic").lsa
    assert lsa.term_correlation("human", "user") == pytest.approx(-0.38, abs=0.01)
    assert lsa.term_correlation("human", "user", k=2) == pytest.approx(0.94, abs=0.01)
    assert lsa.term_correlation("human", "minors") == pytest.approx(-0.29, abs=0.01)
    assert lsa.term_correlation("human", "minors", k=2) == pytest.approx(-0.83, abs=0.01)


def test_lsa_beats_raw_tfidf_on_news():
    ex = run_experiment("news")
    comp = cl.compare(ex.lsa, ex.labels, k=5, n_clusters=4, seeds=(0, 1, 2))
    assert comp.lsa_ari.mean() > comp.raw_ari.mean() + 0.3


def test_query_with_unknown_words_scores_zero():
    ex = run_experiment("news")
    res = ex.search("zzzz qqqq", 5)
    assert res["known_terms"] == []
    assert np.all(res["raw"] == 0) and np.all(res["lsa"] == 0)


def test_custom_corpus_with_labels():
    c = custom_corpus("a | red apples and green apples\nb | the train left the station\na | apples and pears")
    assert c.n_docs == 3 and c.has_labels
    ex = run_experiment(c)
    assert ex.tdm.n_docs == 3
