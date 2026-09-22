"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-22, Toleranzen fangen Rundung ab). `analyse()` und die Experiment-Funktionen sind deterministisch (kein Zufall
außer im Datenerzeuger und der - festen, mit `seed` reproduzierbaren - Teilstichprobe)."""

import functools

import pytest

import lgbm_constants as C
import lgbm_evaluation as ev
import lgbm_scenario as S

PRESET = {"standard": "🌳 Standard", "stump": "🪓 Ein Schritt (kein Boosting)", "coarse": "🔲 Grobe Bins", "fine": "🔬 Feine Bins", "reg": "📈 Regression Standard"}


@functools.lru_cache(maxsize=None)
def _preset(key):
    p = C.PRESETS[PRESET[key]]
    return ev.analyse(p["task"], p["num_leaves"], p["depth"], p["max_bin"], p["lam"], p["gamma"], p["min_child_weight"], p["n_rounds"], p["lr"], p["subsample"],
                      p["n"], p["n_noise"], p["label_noise"], p["seed"])


def _help(key, *needles):
    text = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in text, (key, n)


@functools.lru_cache(maxsize=None)
def _default():
    return ev.analyse("class", C.DEFAULT_NUM_LEAVES, C.DEFAULT_DEPTH, C.DEFAULT_MAX_BIN, C.DEFAULT_LAM, C.DEFAULT_GAMMA, C.DEFAULT_MIN_CHILD_WEIGHT,
                      C.DEFAULT_N_ROUNDS, C.DEFAULT_LR, C.DEFAULT_SUBSAMPLE, C.DEFAULT_N, C.DEFAULT_NOISE, 0, C.DEFAULT_SEED)


# --- Preset-Hilfen --------------------------------------------------------------------------------------------------------------------------------

def test_standard_preset():
    a = _preset("standard")
    assert (a.train["error"], a.test["error"], a.n_leaves) == pytest.approx((0.0607, 0.1667, 975), abs=0.0015)
    _help("standard", "6.1 %", "16.7 %", "975")


def test_single_step_preset_is_no_better_than_a_shallow_stump():
    a = _preset("stump")
    assert len(a.ensemble.trees) == 1 and a.verdict == "stump"
    assert a.test["error"] == pytest.approx(0.1861, abs=0.0015)
    _help("stump", "18.6 %")


def test_coarse_bins_hurt_and_fine_bins_help_slightly():
    coarse = _preset("coarse")
    fine = _preset("fine")
    standard = _preset("standard")
    assert coarse.max_bin == 8 and fine.max_bin == 255
    assert coarse.test["error"] == pytest.approx(0.1972, abs=0.0015)
    assert fine.test["error"] == pytest.approx(0.1611, abs=0.0015)
    assert coarse.test["error"] > standard.test["error"] > fine.test["error"]
    _help("coarse", "19.7 %", "16.7 %")
    _help("fine", "16.1 %", "0.2 %")


def test_regression_standard_preset():
    a = _preset("reg")
    assert a.task == "reg"
    assert (a.test["rmse"], a.n_leaves) == pytest.approx((10.509, 1842), abs=0.02)
    _help("reg", "10.5", "1842")


def test_every_preset_is_a_valid_setting():
    for name, p in C.PRESETS.items():
        assert p["task"] in C.TASKS
        assert C.NUM_LEAVES_MIN <= p["num_leaves"] <= C.NUM_LEAVES_MAX and C.DEPTH_MIN <= p["depth"] <= C.DEPTH_MAX
        assert C.MAX_BIN_MIN <= p["max_bin"] <= C.MAX_BIN_MAX and C.N_ROUNDS_MIN <= p["n_rounds"] <= C.N_ROUNDS_MAX
        assert C.LR_MIN <= p["lr"] <= C.LR_MAX and C.LAM_MIN <= p["lam"] <= C.LAM_MAX and C.GAMMA_MIN <= p["gamma"] <= C.GAMMA_MAX
        assert C.MIN_CHILD_WEIGHT_MIN <= p["min_child_weight"] <= C.MIN_CHILD_WEIGHT_MAX and C.SUBSAMPLE_MIN <= p["subsample"] <= C.SUBSAMPLE_MAX
        d = C.N_BASE + p["n_noise"]
        assert C.N_MIN <= p["n"] <= C.N_MAX and 0 <= p["fx"] < d and 0 <= p["fy"] < d and name in C.PRESET_HELP


# --- Standardansicht --------------------------------------------------------------------------------------------------------------------------------

def test_default_view_numbers():
    a = _default()
    assert (a.train["error"], a.test["error"], a.baseline) == pytest.approx((0.0607, 0.1667, 0.4639), abs=0.0015)


def test_round_curve_reaches_a_minimum():
    a = _default()
    rows = ev.round_rows(a)
    best = ev.best_round(rows)
    assert best["k"] <= a.n_rounds and best["test"] <= rows[-1]["test"] + 1e-9


# --- Blattweise gegen ebenenweise -------------------------------------------------------------------------------------------------------------------

def test_leaf_wise_wins_on_large_clean_data_level_wise_wins_on_small_noisy_data():
    comp = ev.policy_comparison(15, C.DEFAULT_N_ROUNDS, C.DEFAULT_LR, C.DEFAULT_MAX_BIN, 1.0, 0.0, 1.0)
    sn, lc = comp["small_noisy"], comp["large_clean"]
    assert (sn["leaf"], sn["level"]) == pytest.approx((0.1967, 0.1917), abs=0.002)
    assert (lc["leaf"], lc["level"]) == pytest.approx((0.1471, 0.1527), abs=0.002)
    assert sn["level"] < sn["leaf"]                                                                # klein & verrauscht: ebenenweise (leichte Bremse) gewinnt
    assert lc["leaf"] < lc["level"]                                                                # groß & sauber: blattweise gewinnt


# --- Zähler: Histogramme gegen exakte Suche -----------------------------------------------------------------------------------------------------------

def test_histogram_advantage_grows_with_training_set_size():
    rows = ev.counter_rows(C.DEFAULT_NUM_LEAVES, C.DEFAULT_MAX_BIN, C.DEFAULT_NOISE, 0)
    ratios = [r["exact_candidates"] / r["histogram_candidates"] for r in rows]
    assert ratios == pytest.approx([0.694, 0.959, 1.195, 2.293, 3.000], abs=0.02)
    assert all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1))                          # streng monoton wachsend mit der Datenmenge
    assert ratios[0] < 1.0 < ratios[-1]                                                             # bei kleinen Daten ist die Histogramm-Suche sogar TEURER


# --- Grenzfälle --------------------------------------------------------------------------------------------------------------------------------------

def test_generator_matches_cart_demo_conventions():
    ds = S.generate_dataset(500, 3, 0, 7)
    assert ds.X.shape == (500, 11) and ds.names[:2] == ("Distanz", "Ladegewicht")
    Xtr, ytr, Xte, yte = S.split(ds, "class")
    assert len(Xtr) == 350 and len(Xte) == 150
