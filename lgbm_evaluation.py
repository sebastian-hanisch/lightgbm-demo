"""Messungen an LightGBM: Testfehler blatt- gegen ebenenweise bei gleicher Blattzahl auf kleinen/verrauschten gegen großen/sauberen Daten (Überanpassung), Zähler (Histogramme, geprüfte Schnitte)
gegen exakte Suche bei wachsender Datenmenge."""

from dataclasses import dataclass

import numpy as np

import lgbm_algorithm as lgm
import lgbm_constants as C
import lgbm_scenario as S
import lgbm_tree as T


def baseline_error(ds, task):
    _, ytr, _, yte = S.split(ds, task)
    if task == "class":
        return float(np.mean(yte != int(ytr.mean() > 0.5)))
    return float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))


def _metrics(ensemble, X, y, task):
    if task == "class":
        return {"error": float(np.mean(lgm.predict(ensemble, X) != y))}
    v = lgm.predict_value(ensemble, X)
    return {"rmse": float(np.sqrt(np.mean((v - y) ** 2))), "mae": float(np.mean(np.abs(v - y)))}


def primary(metrics, task):
    return metrics["error"] if task == "class" else metrics["rmse"]


def ensemble_importances(ensemble):
    imps = [T.importances(t) for t in ensemble.trees]
    return np.mean(imps, axis=0) if imps else np.zeros(0)


@dataclass
class Analysis:
    ds: object
    task: str
    num_leaves: int
    max_depth: int
    max_bin: int
    lam: float
    gamma: float
    min_child_weight: float
    n_rounds: int
    lr: float
    subsample: float
    ensemble: object
    train: dict
    test: dict
    baseline: float
    verdict: str
    n_leaves: int
    imp: np.ndarray


def analyse(task, num_leaves, max_depth, max_bin, lam, gamma, min_child_weight, n_rounds, lr, subsample, n, n_noise, label_noise, seed):
    ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, seed)
    Xtr, ytr, Xte, yte = S.split(ds, task)
    ensemble = lgm.fit(Xtr, ytr.astype(float), task, num_leaves=num_leaves, max_depth=max_depth, max_bin=max_bin, lam=lam, gamma=gamma,
                       min_child_weight=min_child_weight, n_rounds=n_rounds, learning_rate=lr, subsample=subsample / 100.0, seed=0)
    train = _metrics(ensemble, Xtr, ytr, task)
    test = _metrics(ensemble, Xte, yte, task)
    baseline = baseline_error(ds, task)
    a = Analysis(ds, task, num_leaves, max_depth, max_bin, lam, gamma, min_child_weight, n_rounds, lr, subsample, ensemble, train, test, baseline, "",
                 lgm.n_leaves_total(ensemble), ensemble_importances(ensemble))
    a.verdict = verdict(a)
    return a


def verdict(a):
    tr, te = primary(a.train, a.task), primary(a.test, a.task)
    if a.n_rounds <= 1:
        return "stump"
    over = (te - tr > C.OVERFIT_GAP_CLASS) if a.task == "class" else (te > C.OVERFIT_RATIO_REG * max(tr, 1e-9))
    if over:
        return "overfit"
    return "underfit" if te > C.UNDERFIT_SHARE * a.baseline else "ok"


# --- Testfehler gegen die Rundenzahl ----------------------------------------------------------------------------------------------------------------

def round_rows(a, ks=None):
    ks = ks or sorted(set(np.unique(np.round(np.geomspace(1, a.n_rounds, min(24, a.n_rounds))).astype(int))))
    Xtr, ytr, Xte, yte = S.split(a.ds, a.task)
    rows = []
    for k in ks:
        tr = primary(_metrics_upto(a.ensemble, Xtr, ytr, a.task, k), a.task)
        te = primary(_metrics_upto(a.ensemble, Xte, yte, a.task, k), a.task)
        rows.append({"k": int(k), "train": tr, "test": te})
    return rows


def _metrics_upto(ensemble, X, y, task, upto):
    if task == "class":
        return {"error": float(np.mean(lgm.predict(ensemble, X, upto=upto) != y))}
    v = lgm.predict_value(ensemble, X, upto=upto)
    return {"rmse": float(np.sqrt(np.mean((v - y) ** 2))), "mae": float(np.mean(np.abs(v - y)))}


def best_round(rows):
    return min(rows, key=lambda r: (r["test"], r["k"]))


# --- Blattweise gegen ebenenweise bei gleicher Blattzahl --------------------------------------------------------------------------------------------

def policy_rows(num_leaves, n_rounds, lr, max_bin, lam, gamma, min_child_weight, n, n_noise, label_noise, seeds=C.SWEEP_SEEDS):
    """Testfehler blatt- gegen ebenenweise (`lgbm_tree.grow(..., policy=...)`), gemittelt über mehrere Datensätze, bei sonst identischen Einstellungen (nur die Reihenfolge der Schnitte unterscheidet sich)."""
    rows = {}
    for policy in ("leaf", "level"):
        errs = []
        for sd in seeds:
            ds = S.generate_dataset(n, n_noise, label_noise, sd)
            Xtr, ytr, Xte, yte = S.split(ds, "class")
            ens = lgm.fit(Xtr, ytr.astype(float), "class", num_leaves=num_leaves, max_depth=12, max_bin=max_bin, lam=lam, gamma=gamma,
                          min_child_weight=min_child_weight, n_rounds=n_rounds, learning_rate=lr, subsample=1.0, seed=0, policy=policy)
            errs.append(float(np.mean(lgm.predict(ens, Xte) != yte)))
        rows[policy] = float(np.mean(errs))
    return rows


def policy_comparison(num_leaves, n_rounds, lr, max_bin, lam, gamma, min_child_weight):
    """Die zwei Datenlagen aus dem Plan: klein und verrauscht (kleines n, viel Rauschen/Etiketten-Rauschen) gegen groß und sauber."""
    small = policy_rows(num_leaves, n_rounds, lr, max_bin, lam, gamma, min_child_weight, 400, 6, 10)
    large = policy_rows(num_leaves, n_rounds, lr, max_bin, lam, gamma, min_child_weight, 3000, 3, 0)
    return {"small_noisy": small, "large_clean": large}


# --- Zähler: Histogramme und geprüfte Schnitte gegen exakte Suche ------------------------------------------------------------------------------------

N_GRID = (400, 800, 1200, 2000, 3000)


def counter_rows(num_leaves, max_bin, n_noise, label_noise, seed=C.DEFAULT_SEED, grid=N_GRID):
    """Für eine wachsende Trainingsmenge: geprüfte Schnittkandidaten mit Histogrammen (ein Baum) gegen eine exakte Suche derselben Baumgröße (Summe (Zeilen im Knoten - 1) * Merkmale über alle
    tatsächlichen Schnitte des gewachsenen Baums) - zeigt, dass der Histogramm-Vorteil mit der Datenmenge wächst (Eimerzahl bleibt fest, exakte Schnittsuche wächst mit den Zeilen)."""
    rows = []
    for n in grid:
        ds = S.generate_dataset(n, n_noise, label_noise, seed)
        Xtr, ytr, _, _ = S.split(ds, "class")
        edges = T.build_bin_edges(Xtr, max_bin)
        f0 = lgm.init_value(ytr.astype(float), "class")
        g, h = lgm.grad_hess(ytr.astype(float), np.full(len(ytr), f0), "class")
        stats = {}
        tree = T.grow(Xtr, g, h, edges, num_leaves=num_leaves, max_depth=12, lam=1.0, gamma=0.0, min_child_weight=1.0, min_child_samples=1, policy="leaf", stats=stats)
        d = Xtr.shape[1]
        exact = sum((int(tree.n[t]) - 1) * d for t in tree.internal_nodes())
        rows.append({"n": n, "train_rows": len(ytr), "histogram_candidates": stats["candidates_checked"], "exact_candidates": exact, "leaves": tree.n_leaves})
    return rows
