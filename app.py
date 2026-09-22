"""LightGBM - blattweises Wachsen und Histogramm-Schnittsuche - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - LightGBM - und lässt stattdessen das Beispiel wachsen.
Achtes Stück der Baumbasierten Linie der "Konzepte"-Reihe, viertes Stück des Boosting-Asts (nach AdaBoost, Gradient Boosting, XGBoost): dasselbe regularisierte Ziel wie XGBoost, aber zwei
Änderungen am Baumkern, die reine Rechenzeit sparen - Histogramm-Schnittsuche (Eimer statt jeder einzelnen Schwelle) und blattweises statt ebenenweises Wachsen (immer das Blatt mit dem größten
Gewinn zuerst).
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import lgbm_algorithm as lgm
import lgbm_constants as C
import lgbm_evaluation as ev
from lgbm_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from lgbm_visualization import (
    build_counter_chart,
    build_importance,
    build_map,
    build_policy_chart,
    build_round_curve,
    build_round_tree,
    feature_label,
)

st.set_page_config(page_title="LightGBM – Sebastian Hanisch", layout="wide")

VERDICT_TEXT = {
    "stump": "ℹ️ **Nur ein Schritt eingestellt** - das ist die Vorhersage eines einzelnen (mit Lernrate skalierten) Baums, noch kein Boosting.",
    "overfit": "⚠️ **Überanpassung:** der Testfehler liegt deutlich über dem Trainingsfehler - weniger Blätter oder mehr γ probieren.",
    "underfit": "⚠️ **Unteranpassung:** kaum besser als Raten - mehr Runden, mehr Blätter oder größere Lernrate könnten helfen.",
    "ok": "✅ **Sieht vernünftig aus:** Training und Test liegen nicht weit auseinander.",
}


def _err(task, x):
    return f"{x:.1%}" if task == "class" else f"{x:.1f} min"


@st.cache_resource(show_spinner=False, max_entries=24)
def _analysis(*params):
    return ev.analyse(*params)


@st.cache_data(show_spinner=False, max_entries=8)
def _round_rows(*params):
    a = _analysis(*params)
    return ev.round_rows(a)


@st.cache_data(show_spinner=False, max_entries=4)
def _policy_comparison(num_leaves, n_rounds, lr, max_bin, lam, gamma, mcw):
    return ev.policy_comparison(num_leaves, n_rounds, lr, max_bin, lam, gamma, mcw)


@st.cache_data(show_spinner=False, max_entries=4)
def _counter_rows(num_leaves, max_bin, n_noise, label_noise):
    return ev.counter_rows(num_leaves, max_bin, n_noise, label_noise)


st.title("🍃🌳 LightGBM – blattweises Wachsen und Histogramm-Schnittsuche")
st.markdown(
    """
**XGBoost** (voriges Stück) sucht bei jedem Schnitt die exakt beste Schwelle jedes Merkmals und wächst Ebene für Ebene. **LightGBM** (Ke et al. 2017) übernimmt dasselbe regularisierte Ziel
und dieselbe Gewinnformel - ändert aber, WIE geschnitten und WANN geteilt wird, um Rechenzeit zu sparen: **Histogramm-Schnittsuche** (jedes Merkmal wird vorab in Eimer eingeteilt, nur die
Eimer-Grenzen werden geprüft, nicht jede einzelne Schwelle) und **blattweises Wachsen** (immer das Blatt mit dem größten möglichen Gewinn wird als Nächstes geteilt, statt eine ganze Ebene
abzuarbeiten, bevor die nächste beginnt).
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - achtes Stück der Baumbasierten Linie der \"Konzepte\"-Reihe und viertes "
    "Stück des **Boosting-Asts** (nach AdaBoost, Gradient Boosting, XGBoost) - **ein** Verfahren an einem wachsenden Beispiel. Das Verfahren geht auf Ke et al. (2017) zurück; alle Lieferungen, "
    "Merkmale und Zahlen dieser Demo sind erzeugt und gemessen - keine echten Daten. Der Baumkern (`lgbm_tree.py`) ist neu geschrieben; die echte `lightgbm`-Bibliothek kommt nur in den Tests als "
    "Gegenprobe vor (über Rang-/Fehlergrenzen, nicht exakt - die eigene Eimer-Regel unterscheidet sich von der echten Bibliothek)."
)
st.caption(
    "**Bezug zu OR:** kürzere Rechenzeit je Baum erlaubt mehr Runden oder größere Datensätze im selben Zeitbudget - bei einer täglich neu zu berechnenden Lieferzeitprognose für die Tourenplanung "
    "zählt jede Sekunde Trainingszeit."
)

with st.expander("So funktioniert LightGBM", expanded=True):
    st.markdown(
        r"""
1. **Eimer einmal bilden:** jedes Merkmal wird vor der ersten Runde in `max_bin` Eimer eingeteilt (Quantil-Grenzen über die ganze Trainingsmenge) - dieselben Eimer gelten für jede Runde und jeden Knoten.
2. **Histogramm je Knoten:** für die Zeilen eines Knotens werden Gradient- und Hesse-Summen JE EIMER aufaddiert (ein Histogramm) - die Schnittsuche prüft dann nur noch die Eimer-Grenzen, mit
   derselben Gewinnformel wie XGBoost: $0.5\big[\tfrac{G_L^2}{H_L+\lambda}+\tfrac{G_R^2}{H_R+\lambda}-\tfrac{G^2}{H+\lambda}\big]-\gamma$.
3. **Differenz-Trick:** wird ein Blatt geteilt, wird nur das KLEINERE Kind direkt aus seinen Zeilen histogrammiert - das größere Kind ergibt sich als Elternhistogramm minus kleineres Kind.
4. **Blattweises Wachsen:** eine Prioritätswarteschlange über alle noch teilbaren Blätter, sortiert nach ihrem besten möglichen Gewinn - immer das Blatt mit dem größten Gewinn wird als Nächstes
   geteilt, bis `num_leaves` erreicht ist oder kein Blatt mehr einen positiven Gewinn hat.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    task = st.selectbox("Aufgabe", C.TASKS, key="task_select", format_func=lambda k: C.TASK_LABELS[k])
    num_leaves = st.slider("Blätter je Baum", *bounds("num_leaves_slider"), key="num_leaves_slider", help="Das Wachstumsbudget je Baum - LightGBMs Haupt-Regler (statt Tiefe).")
    depth = st.slider("Tiefenobergrenze", *bounds("depth_slider"), key="depth_slider", help="Zusätzliche Bremse; bei 12 greift praktisch nur noch die Blattzahl.")
    max_bin = st.slider("Eimer je Merkmal", *bounds("max_bin_slider"), key="max_bin_slider", help="Wie fein jedes Merkmal vorab eingeteilt wird - mehr Eimer nähern sich der exakten Schnittsuche an, kosten aber mehr Rechenzeit je Schnitt.")
    n_rounds = st.slider("Zahl der Runden", *bounds("n_rounds_slider"), key="n_rounds_slider")
    lr = st.slider("Lernrate", *bounds("lr_slider"), key="lr_slider", step=0.01, format="%.2f")
    lam = st.slider("λ (L2 auf Blattgewichte)", *bounds("lam_slider"), key="lam_slider", step=0.1, format="%.1f")
    gamma = st.slider("γ (Mindestgewinn je Schnitt)", *bounds("gamma_slider"), key="gamma_slider", step=0.1, format="%.1f")
    mcw = st.slider("Mindest-Hessegewicht je Blatt", *bounds("mcw_slider"), key="mcw_slider", step=0.5, format="%.1f")
    subsample = st.slider("Teilstichprobe je Runde [%]", *bounds("subsample_slider"), key="subsample_slider", format="%.0f%%")
    st.markdown("**Daten**")
    n = st.slider("Lieferungen", *bounds("n_slider"), key="n_slider", step=100)
    n_noise = st.slider("Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider")
    if task == "class":
        label_noise = st.slider("Falsche Etiketten im Training [%]", *bounds("label_noise_slider"), key="label_noise_slider")
        st.session_state["_label_noise_kept"] = label_noise
    else:
        label_noise = int(st.session_state.get("_label_noise_kept", C.DEFAULT_LABEL_NOISE))
        st.caption("Falsche Etiketten gibt es nur bei der Klassifikation.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Daten generieren", width="stretch", on_click=randomize_seed)

base_params = (task, int(num_leaves), int(depth), int(max_bin), float(lam), float(gamma), float(mcw), int(n_rounds), float(lr), int(subsample))
data_params = (int(n), int(n_noise), int(label_noise), int(seed))
with st.spinner("Rechne ..."):
    a = _analysis(*base_params, *data_params)
ds = a.ds
names = ds.names
n_feat = len(names)
n_test = len(ds.test)
n_trees = len(a.ensemble.trees)

with st.sidebar:
    st.markdown("**Ansicht**")
    for key, default in (("map_x_select", C.DEFAULT_MAP[0]), ("map_y_select", C.DEFAULT_MAP[1])):
        if st.session_state[key] >= n_feat:
            st.session_state[key] = default
    fx = st.selectbox("Karte: waagerecht", range(n_feat), key="map_x_select", format_func=lambda f: feature_label(names, f))
    fy = st.selectbox("Karte: senkrecht", range(n_feat), key="map_y_select", format_func=lambda f: feature_label(names, f))
    if st.session_state.get("sample_slider", 0) > n_test - 1:
        st.session_state["sample_slider"] = 0
    sample_idx = st.slider("Testlieferung", 0, n_test - 1, 0, key="sample_slider")
sync_query_params({"task_select": task, "num_leaves_slider": int(num_leaves), "depth_slider": int(depth), "max_bin_slider": int(max_bin), "n_rounds_slider": int(n_rounds),
                   "lr_slider": float(lr), "lam_slider": float(lam), "gamma_slider": float(gamma), "mcw_slider": float(mcw), "subsample_slider": int(subsample),
                   "n_slider": int(n), "n_noise_slider": int(n_noise), "label_noise_slider": int(label_noise), "seed_input": int(seed), "map_x_select": int(fx), "map_y_select": int(fy)})

view_key = (base_params, data_params)
if st.session_state.get("lgbm_owner") != view_key:
    st.session_state["lgbm_owner"] = view_key
    st.session_state["lgbm_step"] = n_trees

# --- LightGBM in Aktion -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🍃 LightGBM in Aktion")
st.caption("Runde für Runde: links der Baum dieser Runde (Zahlen an den Knoten = Wachstumsreihenfolge, 0 zuerst - springt zwischen Ebenen statt sie der Reihe nach zu füllen), rechts die Vorhersage des Ensembles bis dahin.")
if n_trees > 1:
    step_col, play_col = st.columns([5, 2])
    with step_col:
        step = st.slider("Runden", 1, n_trees, key="lgbm_step")
    with play_col:
        auto_play = st.button("▶️ Abspielen", width="stretch")
else:
    step, auto_play = 1, False
    st.info("ℹ️ Nur eine Runde eingestellt - mehr Runden in der Seitenleiste zeigen den Effekt.")
view_slot = st.empty()
Xte_full, yte_full = ds.X[ds.test], ds.y_true[ds.test] if task == "class" else ds.y_reg[ds.test]
sample_x = Xte_full[sample_idx]
sample_y = yte_full[sample_idx]


def _render(current):
    tree = a.ensemble.trees[current - 1]
    with view_slot.container():
        c1, c2 = st.columns([2, 3])
        with c1:
            st.plotly_chart(build_round_tree(tree, names), width="stretch", key=f"tree_chart_{current}")
            st.caption(f"Runde {current}: {tree.n_leaves} Blätter.")
        with c2:
            st.plotly_chart(build_map(a.ensemble, ds, task, fx, fy, upto=current, sample=sample_x), width="stretch", key=f"map_chart_{current}")
        pred_here = lgm.predict_value(a.ensemble, sample_x.reshape(1, -1), upto=current)[0]
        pred_text = f"{pred_here:.0%} zu spät" if task == "class" else f"{pred_here:.1f} min"
        truth_text = ("zu spät" if sample_y == 1 else "pünktlich") if task == "class" else f"{sample_y:.1f} min"
        st.markdown(f"**Testlieferung {sample_idx}:** Vorhersage nach {current} Runden = **{pred_text}**; tatsächlich: **{truth_text}**.")


if auto_play:
    frames = sorted(set(np.unique(np.round(np.linspace(1, n_trees, min(12, n_trees))).astype(int))))
    for kk in frames:
        _render(kk)
        time.sleep(min(0.9, 6.0 / len(frames)))
    step = n_trees
else:
    _render(step)

st.markdown("---")

# --- Was das Ensemble gelernt hat -------------------------------------------------------------------------------------------------------------------

st.markdown("## 📐 Was das Ensemble gelernt hat – und wie gut es auf neuen Lieferungen ist")
rrows = _round_rows(*base_params, *data_params)
best = ev.best_round(rrows)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Runden", n_trees)
m2.metric("Trainingsfehler" if task == "class" else "Trainings-RMSE", _err(task, ev.primary(a.train, task)))
m3.metric("Testfehler" if task == "class" else "Test-RMSE", _err(task, ev.primary(a.test, task)), delta=f"ohne Modell: {_err(task, a.baseline)}", delta_color="off")
m4.metric("Blätter insgesamt", a.n_leaves, delta=f"bester Testfehler bei Runde {best['k']}", delta_color="off")
st.markdown(VERDICT_TEXT[a.verdict])

st.markdown("**Testfehler gegen die Rundenzahl**")
st.plotly_chart(build_round_curve(rrows, task, a.baseline, n_trees, best["k"]), width="stretch", key="round_chart")
st.caption(f"Der Trainingsfehler sinkt fast durchgehend; der Testfehler erreicht sein Minimum bei Runde {best['k']} ({_err(task, best['test'])}).")

st.markdown("**Wichtigkeit je Merkmal**")
st.plotly_chart(build_importance(names, a.imp), width="stretch", key="importance_chart")
st.caption("Gemittelt über alle Bäume des Ensembles: Summe der Schnittgewinne je Merkmal, auf 1 normiert.")

st.markdown("---")

# --- Experimente -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Blattweise gegen ebenenweise bei gleicher Blattzahl")
if st.button("Testfehler blattweise gegen ebenenweise messen, kleine/verrauschte gegen große/saubere Daten (dauert einen Moment)", key="policy_start"):
    st.session_state["policy_on"] = True
if st.session_state.get("policy_on"):
    with st.spinner("Trainiere beide Wachstumsreihenfolgen auf zwei Datenlagen, je fünf Datensätze ..."):
        comp = _policy_comparison(15, int(n_rounds), float(lr), int(max_bin), float(lam), float(gamma), float(mcw))
    st.plotly_chart(build_policy_chart(comp), width="stretch", key="policy_chart")
    sn, lc = comp["small_noisy"], comp["large_clean"]
    st.caption(f"15 Blätter je Baum, sonst Ihre aktuellen Einstellungen, Mittel über fünf Datensätze. Klein & verrauscht (400 Lieferungen, 6 Rauschmerkmale, 10 % falsche Etiketten): ebenenweise "
               f"leicht besser ({sn['level']:.1%} gegen {sn['leaf']:.1%}) - blattweise jagt hier eher dem Rauschen hinterher. Groß & sauber (3000 Lieferungen, 3 Rauschmerkmale, keine falschen "
               f"Etiketten): blattweise gewinnt ({lc['leaf']:.1%} gegen {lc['level']:.1%}) - mit genug sauberen Daten nutzt die freie Wahl des besten Schnitts mehr, als sie schadet.")

st.markdown("---")

st.subheader("🔬 Zähler: Histogramme gegen exakte Suche")
if st.button("Geprüfte Schnittkandidaten gegen die Trainingsmenge messen (dauert einen Moment)", key="counter_start"):
    st.session_state["counter_on"] = True
if st.session_state.get("counter_on"):
    with st.spinner("Baue je Trainingsmenge einen Baum und zähle mit ..."):
        crows = _counter_rows(int(num_leaves), int(max_bin), int(n_noise), int(label_noise))
    st.plotly_chart(build_counter_chart(crows), width="stretch", key="counter_chart")
    small, large = crows[0], crows[-1]
    ratio = large["exact_candidates"] / large["histogram_candidates"]
    st.caption(f"Bei {small['train_rows']} Trainingszeilen prüft die Histogramm-Suche sogar MEHR Kandidaten als eine exakte Suche ({small['histogram_candidates']} gegen "
               f"{small['exact_candidates']}) - die feste Eimerzahl lohnt sich erst, wenn die Knoten mehr Zeilen haben als Eimer. Bei {large['train_rows']} Zeilen dreht sich das Bild: die "
               f"Histogramm-Suche bleibt bei {large['histogram_candidates']} Kandidaten (die Eimerzahl wächst nicht mit der Datenmenge), die exakte Suche bräuchte {large['exact_candidates']} - "
               f"{ratio:.1f}-mal so viele.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Blattweises Wachsen ist immer besser** | Auf kleinen, verrauschten Daten kann blattweises Wachsen dem Rauschen eher folgen als ebenenweises - gemessen oben. | γ (Mindestgewinn) oder weniger Blätter als Bremse |
| **Mehr Eimer sind immer besser** | Mehr Eimer nähern sich der exakten Suche an, kosten aber mehr Rechenzeit je Schnitt - bei sehr kleinen Datensätzen ist der Histogramm-Vorteil sogar negativ (gemessen oben). | Eimerzahl an die Datenmenge anpassen, nicht pauschal maximieren |
| **Exakte Schwellen** | Eimer-Grenzen sind nur Näherungen der besten Schwelle - bei wenigen, groben Eimern kann die wahre beste Schwelle zwischen zwei Eimern verschwinden (siehe "Grobe Eimer"-Beispiel). | mehr Eimer, oder XGBoost/Gradient Boosting bei kleinen Datensätzen |
| **Globale Eimer-Grenzen für alle Runden** | Dieselben Eimer gelten für jede Runde - bei sich stark verändernden Pseudo-Residuen könnten andere Grenzen manche Runden besser bedienen; das echte LightGBM macht das genauso, aus Geschwindigkeitsgründen. | - |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Histogramm-Schnittsuche.** Jedes Merkmal $f$ wird vorab in Eimer $[e_0,e_1),[e_1,e_2),\dots$ eingeteilt (Quantil-Grenzen über die ganze Trainingsmenge). Für einen Knoten mit Zeilen $R$ ist das
Histogramm $H_f[k] = \big(\sum_{i\in R,\,x_{if}\in\text{Eimer }k} g_i,\ \sum_{i\in R,\,x_{if}\in\text{Eimer }k} h_i\big)$ - danach ist die Schnittsuche genau wie in xgboost-demo, nur über die
Eimer-Grenzen statt über jede einzelne Schwelle: $\text{Gewinn} = 0.5\big[\tfrac{G_L^2}{H_L+\lambda}+\tfrac{G_R^2}{H_R+\lambda}-\tfrac{G^2}{H+\lambda}\big]-\gamma$, Blattwert $-\tfrac{G}{H+\lambda}$.

**Differenz-Trick.** Für einen Knoten mit Histogramm $H$, der in L und R geteilt wird: wird L direkt aus seinen Zeilen histogrammiert, ergibt sich $H_R = H - H_L$ ohne weiteren Zeilenzugriff -
LightGBM histogrammiert deshalb immer das KLEINERE Kind direkt.

**Blattweises Wachsen.** Eine Prioritätswarteschlange über alle Blätter mit einem gültigen Schnitt, geordnet nach ihrem Gewinn; wiederholt wird das Blatt mit dem größten Gewinn geteilt, bis
`num_leaves` Blätter erreicht sind oder kein Blatt mehr einen positiven Gewinn hat. Bei GLEICHER Blattzahl ist das Ergebnis nur dann identisch mit ebenenweisem Wachsen, wenn beide Verfahren
ohnehin alle möglichen positiven Schnitte machen (kein bindendes Blattbudget) - sobald das Budget bindet, wählt blattweise die wertvollsten Schnitte zuerst, ebenenweise nimmt sie in
Entstehungsreihenfolge (geprüft in `tests/test_algorithm.py`).

Implementiert in `lgbm_tree.py` (Eimer, Histogramm, Differenz-Trick, blatt-/ebenenweises Wachsen), `lgbm_algorithm.py` (Fit, Vorhersage), `lgbm_evaluation.py` (Analyse, Rundenkurve, die zwei Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
