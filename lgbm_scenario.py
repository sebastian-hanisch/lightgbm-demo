"""Die Lieferdaten der Demo: eigene, erzeugte Lieferungen mit acht echten Merkmalen (Distanz, Ladegewicht, Stopps, Verkehr, Wetter, Wochentag, Zeitfenster, Fahrerjahre) und einstellbar vielen Rauschmerkmalen ohne Bezug zum Ziel.
Ziel der Regression ist die Dauer in Minuten, Ziel der Klassifikation "zu spät" (Dauer über der versprochenen Zeit). Wortgleich aus xgboost-demo übernommen (dort wortgleich aus gradient-boosting-demo)."""

from dataclasses import dataclass

import numpy as np

import lgbm_constants as C

DURATION_NOISE = 7.0
PROMISE_SLACK = 14.0


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y_reg: np.ndarray
    y_cls: np.ndarray
    y_true: np.ndarray
    names: tuple
    train: np.ndarray
    test: np.ndarray
    n_noise: int
    label_noise: float
    seed: int

    @property
    def n(self):
        return len(self.X)

    def y(self, task):
        return self.y_cls if task == "class" else self.y_reg


def _base_columns(n, rng):
    dist = np.clip(rng.gamma(2.2, 22.0, n), 3.0, 160.0)
    weight = rng.uniform(100.0, 1200.0, n)
    stops = np.clip(rng.poisson(12, n), 2, 40).astype(float)
    traffic = rng.beta(2.0, 2.0, n)
    weather = rng.beta(1.5, 3.0, n)
    weekday = rng.integers(0, 7, n).astype(float)
    window = rng.uniform(0.0, 1.0, n)
    years = np.clip(rng.gamma(2.0, 3.5, n), 0.0, 20.0)
    return np.column_stack([dist, weight, stops, traffic, weather, weekday, window, np.floor(years)])


def duration(X):
    dist, weight, stops, traffic, weather, weekday, window, years = (X[:, i] for i in range(8))
    base = 25.0 + 0.5 * dist + 1.2 * stops
    jam = 0.9 * dist * np.clip((traffic - 0.6) / 0.4, 0.0, 1.0)
    load = np.where(weight > 800.0, 1.5 * stops, 0.0)
    rain = np.where(weather > 0.65, 0.25 * dist, 0.0)
    weekend = np.where(weekday >= 5, -8.0, 0.0)
    skill = -0.6 * np.minimum(years, 8.0)
    return base + jam + load + rain + weekend + skill


def promised(X):
    dist, stops, window = X[:, 0], X[:, 2], X[:, 6]
    return 25.0 + 0.5 * dist + 1.2 * stops + PROMISE_SLACK - 10.0 * window


def generate_dataset(n=C.DEFAULT_N, n_noise=C.DEFAULT_NOISE, label_noise=C.DEFAULT_LABEL_NOISE, seed=C.DEFAULT_SEED):
    rng = np.random.default_rng([int(seed), 2024])
    n = int(n)
    Xb = _base_columns(n, rng)
    noise = rng.normal(0.0, 1.0, (n, C.NOISE_MAX))
    dur = duration(Xb) + rng.normal(0.0, DURATION_NOISE, n)
    y_true = (dur > promised(Xb)).astype(int)
    flips = rng.random(n) < float(label_noise) / 100.0
    y_cls = np.where(flips, 1 - y_true, y_true)
    perm = np.random.default_rng([int(seed), 99]).permutation(n)
    cut = int(round(n * (1.0 - C.TEST_SHARE)))
    n_noise = int(n_noise)
    X = np.column_stack([Xb, noise[:, :n_noise]]) if n_noise else Xb
    names = tuple(f[0] for f in C.FEATURES) + tuple(f"Rauschen {i + 1}" for i in range(n_noise))
    return Dataset(X, dur, y_cls, y_true, names, np.sort(perm[:cut]), np.sort(perm[cut:]), n_noise, float(label_noise), int(seed))


def split(ds, task):
    y = ds.y(task)
    y_test = ds.y_true if task == "class" else ds.y_reg
    return ds.X[ds.train], y[ds.train], ds.X[ds.test], y_test[ds.test]
