"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

FEATURES = [("Distanz", "km"), ("Ladegewicht", "kg"), ("Stopps", ""), ("Verkehr", "0-1"), ("Wetter", "0-1"), ("Wochentag", "0 = Mo"), ("Zeitfenster-Enge", "0-1"), ("Fahrerjahre", "Jahre")]
N_BASE = len(FEATURES)

TASKS = ("class", "reg")
TASK_LABELS = {"class": "Klassifikation: kommt die Lieferung zu spät?", "reg": "Regression: wie lange dauert die Lieferung?"}
DEFAULT_TASK = "class"

N_MIN, N_MAX, DEFAULT_N = 400, 3000, 1200
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0, 8, 3
LABEL_NOISE_MIN, LABEL_NOISE_MAX, DEFAULT_LABEL_NOISE = 0, 20, 0
TEST_SHARE = 0.3
DEFAULT_SEED = 7

SWEEP_SEEDS = tuple(range(100000, 100005))

OVERFIT_GAP_CLASS = 0.08
OVERFIT_RATIO_REG = 1.6
UNDERFIT_SHARE = 0.75

# LightGBM-eigene Regler
N_ROUNDS_MIN, N_ROUNDS_MAX, DEFAULT_N_ROUNDS = 1, 300, 60
NUM_LEAVES_MIN, NUM_LEAVES_MAX, DEFAULT_NUM_LEAVES = 2, 200, 31
DEPTH_MIN, DEPTH_MAX, DEFAULT_DEPTH = 1, 12, 12                       # 12 = praktisch unbegrenzt (num_leaves greift zuerst)
MAX_BIN_MIN, MAX_BIN_MAX, DEFAULT_MAX_BIN = 4, 255, 63
LR_MIN, LR_MAX, DEFAULT_LR = 0.02, 1.0, 0.1
LAM_MIN, LAM_MAX, DEFAULT_LAM = 0.0, 20.0, 1.0
GAMMA_MIN, GAMMA_MAX, DEFAULT_GAMMA = 0.0, 20.0, 0.0
MIN_CHILD_WEIGHT_MIN, MIN_CHILD_WEIGHT_MAX, DEFAULT_MIN_CHILD_WEIGHT = 0.0, 50.0, 1.0
SUBSAMPLE_MIN, SUBSAMPLE_MAX, DEFAULT_SUBSAMPLE = 20, 100, 100

DEFAULT_MAP = (0, 3)

COLORS = {"train": "#1f77b4", "test": "#d62728", "leafwise": "#1f77b4", "levelwise": "#d62728", "small": "#2ca02c", "large": "#9467bd"}

PRESETS = {
    "🌳 Standard": dict(task="class", num_leaves=DEFAULT_NUM_LEAVES, depth=DEFAULT_DEPTH, max_bin=DEFAULT_MAX_BIN, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, lam=1.0, gamma=0.0,
                        min_child_weight=1.0, subsample=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🪓 Ein Schritt (kein Boosting)": dict(task="class", num_leaves=4, depth=DEFAULT_DEPTH, max_bin=DEFAULT_MAX_BIN, n_rounds=1, lr=1.0, lam=1.0, gamma=0.0, min_child_weight=1.0,
                                          subsample=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🔲 Grobe Bins": dict(task="class", num_leaves=DEFAULT_NUM_LEAVES, depth=DEFAULT_DEPTH, max_bin=8, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, lam=1.0, gamma=0.0,
                          min_child_weight=1.0, subsample=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🔬 Feine Bins": dict(task="class", num_leaves=DEFAULT_NUM_LEAVES, depth=DEFAULT_DEPTH, max_bin=255, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, lam=1.0, gamma=0.0,
                          min_child_weight=1.0, subsample=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "📈 Regression Standard": dict(task="reg", num_leaves=DEFAULT_NUM_LEAVES, depth=DEFAULT_DEPTH, max_bin=DEFAULT_MAX_BIN, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, lam=1.0, gamma=0.0,
                                   min_child_weight=1.0, subsample=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
}
PRESET_HELP = {
    "🌳 Standard": "31 Blätter, 60 Runden, Lernrate 0.1, 63 Bins je Merkmal, λ = 1: Trainingsfehler 6.1 %, Testfehler 16.7 % (Raten: 46.4 %), 975 Blätter insgesamt über alle Runden.",
    "🪓 Ein Schritt (kein Boosting)": "Ein einzelner Baum mit höchstens 4 Blättern (Lernrate 1, keine weiteren Runden): Testfehler 18.6 % - kaum besser als Raten.",
    "🔲 Grobe Bins": "Nur 8 Bins je Merkmal statt 63: Testfehler steigt auf 19.7 % (Standard: 16.7 %) - zu wenige Bins verwischen die beste Schwelle.",
    "🔬 Feine Bins": "255 statt 63 Bins je Merkmal (nahe an exakter Suche): Testfehler sinkt leicht auf 16.1 %, der Trainingsfehler fast auf 0 % (0.2 %) - mehr Bins nähern sich der exakten Split-Suche an, kosten aber mehr Rechenzeit je Split.",
    "📈 Regression Standard": "31 Blätter, 60 Runden, Lernrate 0.1, Ziel Lieferdauer: Test-RMSE 10.5 Minuten, 1842 Blätter insgesamt.",
}
