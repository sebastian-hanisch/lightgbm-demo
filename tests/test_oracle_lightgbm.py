"""Unabhängige Orakel für den LightGBM-Baumkern (Regressionstest der Orakelprüfung).

1. Baum-Orakel: der gewachsene Baum wird Knoten für Knoten mit Brute-Force-Schleifen über alle Bin-Kanten aller Merkmale (direkt aus den Zeilen, keine Histogramme, kein Differenz-Trick) geprüft:
   Blattwert -G/(H+λ), Gain und Optimalität jedes Splits, Mindestblattgröße/-gewicht, Tiefengrenze, Blattbudget und die Reihenfolge (blattweise: größter verfügbarer Gain zuerst; ebenenweise:
   Breitenreihenfolge). Bei Gleichständen ist jeder maximale Split zulässig.
2. Zähler: die Kandidatenzahlen der Histogramm- und der exakten Suche werden aus der Baumstruktur nachgezählt - beide Suchen müssen in denselben Knoten zählen (auch in den am Ende ungeteilten
   Blättern, deren bester Split für die Reihenfolge bestimmt wurde).
3. Echte `lightgbm`-Bibliothek: mit Kanten zwischen allen verschiedenen Werten (= exakte Suche) stimmen die Vorhersagen auf den Trainingszeilen überein (nicht auf neuen Punkten: bei gleichem
   Gain wählt die Demo die kleinste Schwelle, LightGBM die nächste Bin-Grenze)."""

import numpy as np
import pytest

import lgbm_algorithm as lgm
import lgbm_constants as C
import lgbm_evaluation as ev
import lgbm_scenario as S
import lgbm_tree as T


def _brute_best(X, g, h, edges, lam, gamma, mcw, mcs):
    G, H, m = g.sum(), h.sum(), len(g)
    best = None
    for f in range(X.shape[1]):
        for e in edges[f]:
            left = X[:, f] <= e
            cl = int(left.sum())
            Gl, Hl = g[left].sum(), h[left].sum()
            Gr, Hr = G - Gl, H - Hl
            if Hl < mcw or Hr < mcw or cl < mcs or m - cl < mcs:
                continue
            gain = 0.5 * (Gl ** 2 / (Hl + lam) + Gr ** 2 / (Hr + lam) - G ** 2 / (H + lam)) - gamma
            if gain > 0 and (best is None or gain > best):
                best = gain
    return best


def _parent(tree, u):
    p = np.nonzero((tree.left == u) | (tree.right == u))[0]
    return int(p[0]) if len(p) else -1


def _check_tree(tree, X, g, h, edges, num_leaves, max_depth, lam, gamma, mcw, mcs, policy):
    members = {0: np.arange(len(g))}
    for t in range(tree.n_nodes):
        if tree.feature[t] >= 0:
            idx = members[t]
            left = X[idx, tree.feature[t]] <= tree.threshold[t]
            members[int(tree.left[t])], members[int(tree.right[t])] = idx[left], idx[~left]
    best = {}
    for t in range(tree.n_nodes):
        idx = members[t]
        assert tree.value[t] == pytest.approx(-g[idx].sum() / (h[idx].sum() + lam), rel=1e-9, abs=1e-12)
        assert tree.n[t] == len(idx)
        best[t] = None if (tree.depth[t] >= max_depth or len(idx) < 2 * max(mcs, 1)) else _brute_best(X[idx], g[idx], h[idx], edges, lam, gamma, mcw, mcs)
    inner = sorted(tree.internal_nodes(), key=lambda t: tree.split_order[t])
    assert [int(tree.split_order[t]) for t in inner] == list(range(len(inner)))
    for k, t in enumerate(inner):
        assert best[t] is not None, f"Knoten {t} wurde ohne positiven Split geteilt"
        assert tree.gain[t] == pytest.approx(best[t], rel=1e-9, abs=1e-9), f"Knoten {t}: Split nicht optimal"
        done = set(inner[:k])
        frontier = [u for u in range(tree.n_nodes) if u not in done and (u == 0 or _parent(tree, u) in done) and best[u] is not None]
        if policy == "leaf":
            assert best[t] >= max(best[u] for u in frontier) - 1e-9 * max(1.0, best[t])
        else:
            assert tree.depth[t] == min(int(tree.depth[u]) for u in frontier)
    assert tree.n_leaves <= num_leaves
    if tree.n_leaves < num_leaves:
        assert all(best[u] is None for u in range(tree.n_nodes) if tree.feature[u] < 0), "Blattbudget nicht erschöpft, aber ein Blatt wäre teilbar"
    return best


def test_grown_tree_matches_brute_force_search_order_values_and_constraints():
    rng = np.random.default_rng(21)
    for it in range(60):
        n, d = int(rng.integers(6, 45)), int(rng.integers(1, 4))
        max_bin, num_leaves = int(rng.choice([2, 4, 8, 63])), int(rng.integers(2, 10))
        max_depth, lam, gamma = int(rng.choice([2, 3, 12])), float(rng.choice([0.5, 1.0, 5.0])), float(rng.choice([0.0, 0.01, 0.2]))
        mcw, mcs = float(rng.choice([0.0, 1.0, 3.0])), int(rng.choice([1, 1, 2, 4]))
        policy = "leaf" if it % 3 else "level"
        X = rng.integers(0, 7, (n, d)).astype(float) if it % 2 else rng.normal(size=(n, d))        # ganzzahlig = viele Gleichstände
        g = rng.normal(size=n)
        h = rng.uniform(0.2, 2.0, n) if it % 4 else np.ones(n)
        edges = T.build_bin_edges(X, max_bin)
        tree = T.grow(X, g, h, edges, num_leaves=num_leaves, max_depth=max_depth, lam=lam, gamma=gamma, min_child_weight=mcw, min_child_samples=mcs, policy=policy)
        _check_tree(tree, X, g, h, edges, num_leaves, max_depth, lam, gamma, mcw, mcs, policy)


def test_candidate_counters_are_counted_in_the_same_nodes_for_both_searches():
    for n in (400, 1200):
        ds = S.generate_dataset(n, C.DEFAULT_NOISE, 0, C.DEFAULT_SEED)
        Xtr, ytr, _, _ = S.split(ds, "class")
        edges = T.build_bin_edges(Xtr, C.DEFAULT_MAX_BIN)
        g, h = lgm.grad_hess(ytr.astype(float), np.full(len(ytr), lgm.init_value(ytr.astype(float), "class")), "class")
        stats = {}
        tree = T.grow(Xtr, g, h, edges, num_leaves=C.DEFAULT_NUM_LEAVES, max_depth=12, lam=1.0, gamma=0.0, min_child_weight=1.0, min_child_samples=1, stats=stats)
        evaluated = [t for t in range(tree.n_nodes) if tree.depth[t] < 12 and tree.n[t] >= 2]       # alle Knoten, für die ein bester Split gesucht wurde
        assert stats["candidates_checked"] == len(evaluated) * sum(len(e) for e in edges)
        assert stats["exact_candidates"] == sum((int(tree.n[t]) - 1) * Xtr.shape[1] for t in evaluated)
    rows = ev.counter_rows(C.DEFAULT_NUM_LEAVES, C.DEFAULT_MAX_BIN, C.DEFAULT_NOISE, 0, grid=(400,))
    assert (rows[0]["histogram_candidates"], rows[0]["exact_candidates"]) == (21138, 17303)                     # von Hand nachgezählt (siehe oben)


def test_predictions_on_training_rows_match_the_lightgbm_library_for_exact_bins():
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(31)
    for it in range(8):
        n, d = int(rng.integers(60, 200)), int(rng.integers(1, 4))
        num_leaves, rounds, lr = int(rng.choice([2, 4, 8])), int(rng.integers(1, 5)), float(rng.choice([0.1, 0.5]))
        X = rng.normal(size=(n, d))
        y = 2 * X[:, 0] + np.sin(3 * X[:, -1]) + rng.normal(0, 0.4, n)
        edges = [(np.unique(X[:, f])[:-1] + np.unique(X[:, f])[1:]) / 2.0 for f in range(d)]          # eine Kante zwischen je zwei verschiedenen Werten
        f0 = lgm.init_value(y, "reg")
        F = np.full(n, f0)
        for _ in range(rounds):
            g, h = lgm.grad_hess(y, F, "reg")
            F = F + lr * T.predict_value(T.grow(X, g, h, edges, num_leaves=num_leaves, lam=1.0, gamma=0.0, min_child_weight=1e-3, min_child_samples=1), X)
        ref = lgb.LGBMRegressor(n_estimators=rounds, num_leaves=num_leaves, max_depth=-1, learning_rate=lr, reg_lambda=1.0, min_child_samples=1, min_child_weight=1e-3,
                                max_bin=1000, min_data_in_bin=1, verbosity=-1, n_jobs=1).fit(X, y)
        assert np.max(np.abs(F - ref.predict(X))) < 1e-5
