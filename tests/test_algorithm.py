"""LightGBM gegen unabhängige Referenzen: die Histogramm-Split-Suche gegen Brute-Force über dieselben Bin-Grenzen, der Differenz-Trick (größeres Kind = Elternhistogramm minus kleineres Kind) gegen
direkte Neuberechnung, blattweises Wachsen erzeugt bei GLEICHER Blattzahl eine strukturell andere (unregelmäßigere) Baumform als ebenenweises Wachsen (xgboost-demo), Vorhersagen über Rang-/Fehlergrenzen
gegen die echte `lightgbm`-Bibliothek (die eigene Binning-Regel unterscheidet sich - kein exakter Abgleich, siehe README)."""

import sys
from pathlib import Path

import numpy as np
import pytest

import lgbm_algorithm as lgm
import lgbm_scenario as S
import lgbm_tree as T

XGB_DIR = Path(__file__).resolve().parents[2] / "xgboost-demo"


def _reg(n=400, d=5, seed=0, noise=0.4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + rng.normal(0, noise, n)
    return X, y


def _cls(n=400, d=5, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * np.sin(X[:, 1]) + rng.normal(0, noise, n) > 0).astype(float)
    return X, y


# --- Histogramm-Split-Suche gegen Brute-Force über dieselben Bin-Grenzen --------------------------------------------------------------------------

def test_histogram_split_matches_brute_force_over_the_same_bin_edges():
    rng = np.random.default_rng(1)
    n, d = 300, 4
    X = rng.normal(size=(n, d))
    grad = rng.normal(size=n)
    hess = rng.uniform(0.2, 2.0, n)
    edges = T.build_bin_edges(X, max_bin=16)
    bins = T.digitize(X, edges)
    n_bins = [len(e) + 1 for e in edges]
    hist = [T.histogram(bins[:, f], grad, hess, n_bins[f]) for f in range(d)]
    hg, hh, hc = [h[0] for h in hist], [h[1] for h in hist], [h[2] for h in hist]
    lam, gamma = 1.0, 0.0
    f, thr, gain = T.best_split_from_histograms(hg, hh, hc, edges, lam, gamma, 0.0, 1)

    Gtot, Htot = grad.sum(), hess.sum()
    best = None
    for ff in range(d):
        for edge in edges[ff]:
            left = X[:, ff] <= edge
            Gl, Hl = grad[left].sum(), hess[left].sum()
            Gr, Hr = Gtot - Gl, Htot - Hl
            g = 0.5 * (Gl ** 2 / (Hl + lam) + Gr ** 2 / (Hr + lam) - Gtot ** 2 / (Htot + lam)) - gamma
            if best is None or g > best[2]:
                best = (ff, edge, g)
    assert (f, thr) == (best[0], pytest.approx(best[1])) and gain == pytest.approx(best[2], abs=1e-9)


# --- Differenz-Trick gegen direkte Neuberechnung ------------------------------------------------------------------------------------------------------

def test_sibling_difference_trick_matches_direct_recomputation():
    rng = np.random.default_rng(2)
    n, d = 400, 5
    X = rng.normal(size=(n, d))
    grad = rng.normal(size=n)
    hess = rng.uniform(0.5, 1.5, n)
    edges = T.build_bin_edges(X, max_bin=32)
    bins = T.digitize(X, edges)
    n_bins = [len(e) + 1 for e in edges]
    idx = np.arange(n)
    hist_parent = [T.histogram(bins[idx, f], grad[idx], hess[idx], n_bins[f])[:2] for f in range(d)]
    go_left = X[:, 0] <= edges[0][len(edges[0]) // 2]
    idx_l, idx_r = idx[go_left], idx[~go_left]
    hist_l_direct = [T.histogram(bins[idx_l, f], grad[idx_l], hess[idx_l], n_bins[f])[:2] for f in range(d)]
    hist_r_via_diff = [(pg - lg, ph - lh) for (pg, ph), (lg, lh) in zip(hist_parent, hist_l_direct)]
    hist_r_direct = [T.histogram(bins[idx_r, f], grad[idx_r], hess[idx_r], n_bins[f])[:2] for f in range(d)]
    for f in range(d):
        assert np.allclose(hist_r_via_diff[f][0], hist_r_direct[f][0])
        assert np.allclose(hist_r_via_diff[f][1], hist_r_direct[f][1])


# --- Blattweise != Ebenenweise bei gleicher Blattzahl -----------------------------------------------------------------------------------------------

def test_leaf_wise_growth_differs_structurally_from_level_wise_at_the_same_leaf_count():
    if not XGB_DIR.exists():
        pytest.skip("xgboost-demo nicht neben diesem Repo gefunden")
    sys.path.insert(0, str(XGB_DIR))
    import xgb_tree as XT

    rng = np.random.default_rng(0)
    n, d = 400, 5
    X = rng.normal(size=(n, d))
    grad = rng.normal(size=n)
    hess = rng.uniform(0.5, 1.5, n)
    edges = T.build_bin_edges(X, max_bin=255)                                # feine Bins ~ exakte Suche, damit nur das Wachstumsmuster den Unterschied macht
    leafwise = T.grow(X, grad, hess, edges, num_leaves=13, max_depth=12, lam=1.0, gamma=0.0, min_child_weight=0.0, min_child_samples=1)
    levelwise = XT.grow(X, grad, hess, max_depth=4, lam=1.0, gamma=0.0, min_child_weight=0.0)
    assert leafwise.n_leaves == levelwise.n_leaves == 13
    leaf_depths_lw = sorted(set(int(d_) for d_ in leafwise.depth[leafwise.feature < 0]))
    leaf_depths_vw = sorted(set(int(d_) for d_ in levelwise.depth[levelwise.feature < 0]))
    assert leaf_depths_lw != leaf_depths_vw
    assert max(leaf_depths_lw) > max(leaf_depths_vw)                          # blattweise wird an EINEM Ast viel tiefer, ebenenweise bleibt flach und breit
    assert len(leaf_depths_vw) <= 2                                            # level-wise: Blätter höchstens auf zwei benachbarten Ebenen (letzte Ebene unvollständig)


def test_leaf_wise_growth_always_picks_the_single_best_available_gain():
    rng = np.random.default_rng(3)
    n, d = 300, 4
    X = rng.normal(size=(n, d))
    grad = rng.normal(size=n)
    hess = rng.uniform(0.5, 1.5, n)
    edges = T.build_bin_edges(X, max_bin=32)
    tree = T.grow(X, grad, hess, edges, num_leaves=10, max_depth=12, lam=1.0, gamma=0.0, min_child_weight=0.0, min_child_samples=1)
    gains_in_order = [tree.gain[t] for t in sorted(tree.internal_nodes(), key=lambda t: tree.split_order[t])]
    assert all(g >= -1e-9 for g in gains_in_order)                             # jeder gewählte Split hatte zur Zeit seiner Wahl positiven Gain


# --- Fast exakt / über Toleranz gegen die echte lightgbm-Bibliothek ---------------------------------------------------------------------------------

def test_regression_is_close_to_the_lightgbm_library():
    import lightgbm as lgb

    X, y = _reg(600, 5, 0)
    Xt, yt_true = _reg(100, 5, 1, noise=0.0)
    ens = lgm.fit(X, y, "reg", num_leaves=15, max_depth=12, max_bin=63, lam=1.0, gamma=0.0, min_child_weight=1.0, min_child_samples=1,
                  n_rounds=60, learning_rate=0.1, subsample=1.0, seed=0)
    ref = lgb.LGBMRegressor(n_estimators=60, num_leaves=15, max_depth=-1, learning_rate=0.1, reg_lambda=1.0, min_child_weight=1.0,
                             min_child_samples=1, max_bin=63, subsample=1.0, subsample_freq=1, colsample_bytree=1.0, verbosity=-1, random_state=0).fit(X, y)
    p_own = lgm.predict_value(ens, Xt)
    p_ref = ref.predict(Xt)
    assert np.corrcoef(p_own, p_ref)[0, 1] > 0.98
    rmse_own = float(np.sqrt(np.mean((p_own - yt_true) ** 2)))
    rmse_ref = float(np.sqrt(np.mean((p_ref - yt_true) ** 2)))
    assert abs(rmse_own - rmse_ref) < 0.2 * max(rmse_own, rmse_ref)


# --- Grenzfälle --------------------------------------------------------------------------------------------------------------------------------------

def test_min_child_samples_is_respected():
    X, y = _reg(300, 4, 1)
    ens = lgm.fit(X, y, "reg", num_leaves=40, max_depth=12, lam=1.0, min_child_samples=20, n_rounds=5, learning_rate=0.2, seed=0)
    for tree in ens.trees:
        leaves = tree.feature < 0
        assert (tree.n[leaves] >= 20).all()


def test_larger_gamma_prunes_more_aggressively():
    X, y = _reg(400, 5, 2)
    leaves_low = sum(t.n_leaves for t in lgm.fit(X, y, "reg", num_leaves=60, max_depth=12, lam=1.0, gamma=0.0, n_rounds=15, learning_rate=0.2, seed=0).trees)
    leaves_high = sum(t.n_leaves for t in lgm.fit(X, y, "reg", num_leaves=60, max_depth=12, lam=1.0, gamma=10.0, n_rounds=15, learning_rate=0.2, seed=0).trees)
    assert leaves_high < leaves_low


def test_a_single_round_equals_f0_plus_learning_rate_times_the_first_tree():
    X, y = _reg(200, 4, 0)
    ens = lgm.fit(X, y, "reg", num_leaves=15, max_depth=12, lam=1.0, n_rounds=1, learning_rate=0.5, seed=0)
    expected = ens.f0 + 0.5 * T.predict_value(ens.trees[0], X)
    assert np.allclose(lgm.predict_value(ens, X), expected)


def test_subsample_is_reproducible_with_the_same_seed_and_varies_with_a_different_one():
    X, y = _reg(400, 5, 2)
    ens1 = lgm.fit(X, y, "reg", num_leaves=15, max_depth=12, lam=1.0, n_rounds=20, learning_rate=0.2, subsample=0.5, seed=7)
    ens2 = lgm.fit(X, y, "reg", num_leaves=15, max_depth=12, lam=1.0, n_rounds=20, learning_rate=0.2, subsample=0.5, seed=7)
    ens3 = lgm.fit(X, y, "reg", num_leaves=15, max_depth=12, lam=1.0, n_rounds=20, learning_rate=0.2, subsample=0.5, seed=8)
    assert np.array_equal(lgm.predict_value(ens1, X), lgm.predict_value(ens2, X))
    assert not np.array_equal(lgm.predict_value(ens1, X), lgm.predict_value(ens3, X))


def test_generator_matches_cart_demo_conventions():
    ds = S.generate_dataset(500, 3, 0, 7)
    assert ds.X.shape == (500, 11) and ds.names[:2] == ("Distanz", "Ladegewicht")
    Xtr, ytr, Xte, yte = S.split(ds, "class")
    assert len(Xtr) == 350 and len(Xte) == 150


def _old_digitize(X, edges):
    bins = np.empty(X.shape, dtype=np.int32)
    for f in range(X.shape[1]):
        bins[:, f] = np.searchsorted(edges[f], X[:, f], side="right")
    return bins


def test_rare_binary_feature_is_splittable():
    # Bin k = (edges[k-1], edges[k]] muss zu "X <= threshold geht nach links" passen: sonst schickt die Schwelle 1.0 eines 0/1-Merkmals alle Zeilen nach links
    rng = np.random.default_rng(0)
    x = np.zeros(1000)
    x[:30] = 1.0
    X = x.reshape(-1, 1)
    y = 5.0 * x + rng.normal(0, 0.1, 1000)
    edges = T.build_bin_edges(X, 63)
    assert np.array_equal(edges[0], [0.0, 1.0])
    tree = T.grow(X, -y, np.ones(1000), edges, num_leaves=2, min_child_samples=1)
    assert tree.n_leaves == 2 and tree.threshold[0] == 0.0
    pred = T.predict_value(tree, X)
    assert pred[x == 1].mean() > 4.0 and abs(pred[x == 0].mean()) < 0.1


def test_bins_are_consistent_with_the_threshold_rule():
    X = np.array([[0.0], [1.0], [1.0], [2.0], [3.0], [3.0]])
    edges = [np.array([1.0, 3.0])]
    assert T.digitize(X, edges)[:, 0].tolist() == [0, 0, 0, 1, 1, 1]


def test_continuous_data_gives_the_same_trees_as_the_old_binning(monkeypatch):
    # n = 401: die Quantil-Positionen 400*k/63 sind nie ganzzahlig, keine Kante trifft einen Datenwert - alte und neue Zuordnung stimmen überein
    X, y = _reg(401, 5, 3)
    ens_new = lgm.fit(X, y, "reg", num_leaves=15, max_depth=12, lam=1.0, n_rounds=15, learning_rate=0.2)
    monkeypatch.setattr(T, "digitize", _old_digitize)
    ens_old = lgm.fit(X, y, "reg", num_leaves=15, max_depth=12, lam=1.0, n_rounds=15, learning_rate=0.2)
    monkeypatch.undo()
    assert np.array_equal(lgm.predict_value(ens_new, X), lgm.predict_value(ens_old, X))
