"""Plotly-Darstellungen: Entscheidungsgrenze/Regressionsfläche, ein einzelner Runden-Baum (mit Wachstumsreihenfolge markiert), Fehlerkurve gegen Runden, blattweise gegen ebenenweise bei kleinen/
verrauschten gegen großen/sauberen Daten, Zähler-Vergleich (Histogramme gegen exakte Suche), Wichtigkeit. Alle Achsen sind gesperrt (Touch-Scrollen)."""

import numpy as np
import plotly.graph_objects as go

import lgbm_algorithm as lgm
import lgbm_constants as C

CLASS_SCALE = [[0.0, "#2ca02c"], [0.5, "#f2e394"], [1.0, "#d62728"]]
REG_SCALE = "Viridis"


def lock_axes(fig, height=None, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=height, dragmode=False, **layout)
    return fig


def feature_label(names, f):
    unit = dict(C.FEATURES).get(names[f], "")
    return f"{names[f]} [{unit}]" if unit else names[f]


# --- Karte -------------------------------------------------------------------------------------------------------------------------------------------

def build_map(ensemble, ds, task, fx, fy, upto=None, sample=None, height=430):
    Xtr = ds.X[ds.train]
    ytr = ds.y(task)[ds.train]
    med = np.median(Xtr, axis=0)
    gx = np.round(np.linspace(Xtr[:, fx].min(), Xtr[:, fx].max(), 60), 4)
    gy = np.round(np.linspace(Xtr[:, fy].min(), Xtr[:, fy].max(), 60), 4)
    XX, YY = np.meshgrid(gx, gy)
    grid = np.tile(med, (XX.size, 1))
    grid[:, fx], grid[:, fy] = XX.ravel(), YY.ravel()
    z = np.round(lgm.predict_value(ensemble, grid, upto), 3).reshape(XX.shape)
    vmin, vmax = (0.0, 1.0) if task == "class" else (float(np.min(ytr)), float(np.max(ytr)))
    scale = CLASS_SCALE if task == "class" else REG_SCALE
    fig = go.Figure(go.Heatmap(x=gx, y=gy, z=z, colorscale=scale, zmin=vmin, zmax=vmax, opacity=0.55, showscale=False, hovertemplate="%{z:.2f}<extra></extra>"))
    if task == "class":
        fig.add_trace(go.Contour(x=gx, y=gy, z=z, contours=dict(start=0.5, end=0.5, size=1, coloring="none"), line=dict(color="#111111", width=2), showscale=False, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=np.round(Xtr[:, fx], 4), y=np.round(Xtr[:, fy], 4), mode="markers", marker=dict(size=5, color=ytr, colorscale=scale, cmin=vmin, cmax=vmax, line=dict(color="#333333", width=0.5)),
                             hovertemplate="%{x:.3g} / %{y:.3g}<extra></extra>", showlegend=False))
    if sample is not None:
        fig.add_trace(go.Scatter(x=[sample[fx]], y=[sample[fy]], mode="markers", marker=dict(symbol="star", size=16, color="#ffffff", line=dict(color="#111111", width=2)), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title=feature_label(ds.names, fx))
    fig.update_yaxes(title=feature_label(ds.names, fy))
    return lock_axes(fig, height)


# --- Ein einzelner Runden-Baum (Wachstumsreihenfolge markiert) --------------------------------------------------------------------------------------

def tree_layout(tree):
    x = np.zeros(tree.n_nodes)
    counter = 0
    stack = [(0, False)]
    while stack:
        t, done = stack.pop()
        if tree.feature[t] < 0:
            x[t] = counter
            counter += 1
        elif done:
            x[t] = (x[tree.left[t]] + x[tree.right[t]]) / 2.0
        else:
            stack += [(t, True), (int(tree.right[t]), False), (int(tree.left[t]), False)]
    return x, -tree.depth.astype(float)


def build_round_tree(tree, names, height=300):
    """Der Baum einer Runde: Zahlen an den Knoten sind die WACHSTUMSREIHENFOLGE (0 = zuerst geteilt) - blattweise springt dabei zwischen Ebenen, statt sie der Reihe nach zu füllen."""
    x, y = tree_layout(tree)
    inner = tree.feature >= 0
    fig = go.Figure()
    ex, ey = [], []
    for t in np.nonzero(inner)[0]:
        for c in (tree.left[t], tree.right[t]):
            ex += [x[t], x[c], None]
            ey += [y[t], y[c], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#9aa0a6", width=1), hoverinfo="skip", showlegend=False))
    vmax = float(np.max(np.abs(tree.value[~inner]))) if (~inner).any() else 1.0
    size = 12 + 8 * tree.depth.max() - 5 * tree.depth
    color = np.where(inner, np.nan, tree.value)
    fig.add_trace(go.Scatter(x=x[~inner], y=y[~inner], mode="markers+text", text=[f"{tree.value[t]:+.2g}" for t in np.nonzero(~inner)[0]], textposition="bottom center", textfont=dict(size=9),
                             marker=dict(size=size[~inner], color=color[~inner], colorscale="RdBu_r", cmin=-vmax, cmax=vmax, line=dict(color="#111111", width=1)),
                             hovertext=[f"Blatt: Gewicht {tree.value[t]:+.4g}, n={tree.n[t]}" for t in np.nonzero(~inner)[0]], hoverinfo="text", showlegend=False))
    inner_labels = [str(int(tree.split_order[t])) for t in np.nonzero(inner)[0]]
    fig.add_trace(go.Scatter(x=x[inner], y=y[inner], mode="markers+text", text=inner_labels, textposition="middle center", textfont=dict(size=9, color="#ffffff"),
                             marker=dict(size=size[inner] * 0.8, color="#555555"),
                             hovertext=[f"{names[tree.feature[t]]} ≤ {tree.threshold[t]:.4g}? Gain {tree.gain[t]:.4f}, {int(tree.split_order[t]) + 1}. Split" for t in np.nonzero(inner)[0]],
                             hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return lock_axes(fig, height, plot_bgcolor="rgba(0,0,0,0)")


# --- Fehlerkurve gegen Runden --------------------------------------------------------------------------------------------------------------------------

def _error_axis(fig, task):
    fig.update_yaxes(title="Fehlerquote" if task == "class" else "RMSE [min]", rangemode="tozero", **({"tickformat": ".0%"} if task == "class" else {}))


def build_round_curve(rows, task, baseline, current_k, best_k=None, height=340):
    k = [r["k"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=[r["train"] for r in rows], mode="lines+markers", name="Training", line=dict(color=C.COLORS["train"])))
    fig.add_trace(go.Scatter(x=k, y=[r["test"] for r in rows], mode="lines+markers", name="Test", line=dict(color=C.COLORS["test"])))
    fig.add_hline(y=baseline, line=dict(color="#888888", dash="dash"), annotation_text="ohne Modell (Raten)", annotation_position="top right")
    fig.add_vline(x=current_k, line=dict(color="#111111", dash="dot"))
    if best_k is not None and best_k != current_k:
        fig.add_vline(x=best_k, line=dict(color="#2ca02c", dash="dot"), annotation_text="bester Testfehler", annotation_position="bottom left")
    fig.update_xaxes(title="Runden", type="log" if k[-1] > 30 else "linear")
    _error_axis(fig, task)
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


# --- Blattweise gegen ebenenweise --------------------------------------------------------------------------------------------------------------------

def build_policy_chart(comparison, height=340):
    """comparison: {"small_noisy": {"leaf":.., "level":..}, "large_clean": {...}}"""
    groups = [("klein & verrauscht", "small_noisy"), ("groß & sauber", "large_clean")]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[g[0] for g in groups], y=[comparison[g[1]]["leaf"] for g in groups], name="Blattweise", marker_color=C.COLORS["leafwise"],
                         text=[f"{comparison[g[1]]['leaf']:.1%}" for g in groups], textposition="outside", cliponaxis=False))
    fig.add_trace(go.Bar(x=[g[0] for g in groups], y=[comparison[g[1]]["level"] for g in groups], name="Ebenenweise", marker_color=C.COLORS["levelwise"],
                         text=[f"{comparison[g[1]]['level']:.1%}" for g in groups], textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="Testfehler", tickformat=".0%", rangemode="tozero")
    return lock_axes(fig, height, barmode="group", legend=dict(orientation="h", y=1.12))


# --- Zähler: Histogramme gegen exakte Suche --------------------------------------------------------------------------------------------------------

def build_counter_chart(rows, height=340):
    n = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n, y=[r["histogram_candidates"] for r in rows], mode="lines+markers", name="Histogramm (Bins)", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=n, y=[r["exact_candidates"] for r in rows], mode="lines+markers", name="Exakte Suche (alle Werte)", line=dict(color="#d62728")))
    fig.update_xaxes(title="Trainingszeilen")
    fig.update_yaxes(title="Geprüfte Split-Kandidaten (ein Baum)", rangemode="tozero")
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


# --- Wichtigkeit ---------------------------------------------------------------------------------------------------------------------------------------

def build_importance(names, imp, height=330):
    order = np.argsort(-imp, kind="stable")
    colors = ["#ff7f0e" if f >= C.N_BASE else "#1f77b4" for f in order]
    fig = go.Figure(go.Bar(x=imp[order], y=[names[f] for f in order], orientation="h", marker_color=colors, text=[f"{imp[f]:.1%}" for f in order], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Wichtigkeit (Anteil am Gesamtgewinn)", tickformat=".0%", rangemode="tozero")
    return lock_axes(fig, height, showlegend=False).update_layout(margin=dict(l=10, r=60, t=30, b=10))
