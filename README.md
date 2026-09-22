# LightGBM – blattweises Wachsen und Histogramm-Split-Suche – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lightgbm-demo.streamlit.app/)**

Achtes Stück der **Baumbasierten Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning" und das **vierte Stück des Boosting-Asts**
(nach AdaBoost, Gradient Boosting, XGBoost): anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **LightGBM**
(Ke et al. 2017) – an einem wachsenden Beispiel.
Vehikel: dieselben **Lieferungen** wie in cart-demo/.../xgboost-demo, beide Aufgaben (Klassifikation und Regression).
Alle Daten sind erzeugt, alle Zahlen gemessen und in `tests/test_claims.py` festgehalten – keine echten Daten, `lightgbm` nur in den Tests als Gegenprobe.

**Bezug zu OR:** kürzere Rechenzeit je Baum erlaubt mehr Runden oder größere Datensätze im selben Zeitbudget – bei einer täglich neu zu berechnenden Lieferzeitprognose für die Tourenplanung
zählt jede Sekunde Trainingszeit.

**Einordnung in die Reihe:** XGBoost (voriges Stück) sucht bei jedem Split die exakt beste Schwelle jedes Merkmals und wächst Ebene für Ebene. LightGBM übernimmt **dasselbe regularisierte
Ziel und dieselbe Gain-Formel** – ändert aber, WIE geschnitten und WANN geteilt wird, um Rechenzeit zu sparen: **Histogramm-Split-Suche** (Bins statt jeder einzelnen Schwelle) und
**blattweises Wachsen** (immer das Blatt mit dem größten Gain zuerst, statt Ebene für Ebene).

```
CART → Bagging → Random Forest → Extra Trees   (Bagging-Ast, fertig)
CART → AdaBoost → Gradient Boosting → XGBoost → LightGBM (dieses Stück) → CatBoost   (Boosting-Ast)
```

| Frage | Ergebnis (1200 Lieferungen, 3 Rauschmerkmale, 70 % Training / 30 % Test, Seed 7, sofern nicht anders angegeben) |
|---|---|
| **Histogramm-Split-Suche gegen Brute-Force** | ✅ Über dieselben Bin-Grenzen liefert die vektorisierte Histogramm-Suche exakt denselben besten Split wie eine Brute-Force-Suche. |
| **Differenz-Trick gegen direkte Neuberechnung** | ✅ Das größere Kind aus `Elternhistogramm − kleineres Kind` stimmt exakt mit einer direkten Neuberechnung aus seinen eigenen Zeilen überein. |
| **Blattweise ≠ Ebenenweise bei gleicher Blattzahl** | ✅ Auf denselben Histogrammen gewachsen (nur die Reihenfolge unterscheidet sich): bei 13 Blättern erreicht blattweises Wachsen Tiefen von 1 bis 8, ebenenweises bleibt auf 2 bis 4 – strukturell verschiedene Bäume, gleiche Blattzahl. |
| **Kreuzprobe mit der echten `lightgbm`-Bibliothek** | ⚠️ Nur über Rang-/Fehlergrenzen (Korrelation > 0,98, RMSE-Abweichung < 20 %) – die eigene Quantil-Bin-Regel unterscheidet sich von der echten Bibliothek, kein exakter Abgleich. |
| **Testfehler blatt- gegen ebenenweise, klein & verrauscht** (400 Lieferungen, 6 Rauschmerkmale, 10 % falsche Etiketten, 15 Blätter, Mittel über fünf Datensätze) | ⚠️ Ebenenweise leicht besser (19,2 % gegen 19,7 %) – blattweises Wachsen jagt hier eher dem Rauschen hinterher. |
| **Testfehler blatt- gegen ebenenweise, groß & sauber** (3000 Lieferungen, 3 Rauschmerkmale, keine falschen Etiketten, 15 Blätter, Mittel über fünf Datensätze) | ✅ Blattweise gewinnt (14,7 % gegen 15,3 %) – mit genug sauberen Daten nutzt die freie Wahl des besten Splits mehr, als sie schadet. |
| **Zähler: Histogramme gegen exakte Suche** (31 Blätter, 63 Bins, wachsende Trainingsmenge) | ⚠️ Bei 280 Trainingszeilen prüft die Histogramm-Suche sogar MEHR Kandidaten als eine exakte Suche (21 138 gegen 14 663) – der Vorteil kippt erst bei rund 550–600 Zeilen. Bei 2100 Zeilen ist die exakte Suche schon 3,0-mal so teuer wie die Histogramm-Suche (deren Bin-Zahl nicht mit der Datenmenge wächst). |

## Was die Demo zeigt

- **LightGBM in Aktion:** Runde für Runde mit Schritt-Regler und Abspielen: links der Baum dieser Runde (Zahlen an den Knoten = Wachstumsreihenfolge, springt zwischen Ebenen), rechts die
  Vorhersage des Ensembles bis dahin (Entscheidungsgrenze bei Klassifikation, Regressionsfläche bei Regression).
- **Was das Ensemble gelernt hat:** Trainings- und Testfehler gegen die Rundenzahl mit dem besten Testpunkt markiert, Gesamtzahl der Blätter, Wichtigkeit je Merkmal (Summe der Split-Gains).
- **Regler:** Aufgabe, Blätter je Baum (`num_leaves`, LightGBMs Haupt-Regler statt Tiefe), Tiefenobergrenze, Bins je Merkmal, Rundenzahl, Lernrate, λ, γ, Mindest-Hessegewicht, Teilstichprobe,
  Rauschmerkmale, falsche Etiketten, Lieferungen, Seed.
- **Zwei Experimente auf Knopfdruck:** blattweise gegen ebenenweise bei gleicher Blattzahl (klein & verrauscht gegen groß & sauber), Zähler der geprüften Split-Kandidaten gegen die
  Trainingsmenge (Histogramm gegen exakte Suche).

## Modell und Verfahren

- **Baumkern** (`lgbm_tree.py`, neu geschrieben): globale Quantil-Bins je Merkmal (einmal berechnet, für alle Runden und Knoten wiederverwendet); pro Knoten ein Histogramm (Gradient-/Hesse-/
  Zählsumme je Bin); dieselbe regularisierte Gain-Formel wie xgboost-demo, nur über Bin-Grenzen statt Rohwerte. **Differenz-Trick:** beim Teilen wird nur das kleinere Kind direkt
  histogrammiert, das größere per Subtraktion vom Elternhistogramm. **Blattweises Wachsen:** eine Prioritätswarteschlange über alle teilbaren Blätter, sortiert nach ihrem besten Gain.
- Der Kern unterstützt intern auch `policy="level"` (dieselben Histogramme, aber ebenenweise/FIFO statt blattweise/Prioritätswarteschlange) – ausschließlich fürs Experiment, um GENAU EINEN
  Faktor (die Reihenfolge) zu isolieren, statt blattweises Wachsen mit dem generellen Histogramm-gegen-exakt-Unterschied zu vermischen.

## Was nicht funktioniert hat / gefundene Überraschung

- **Ein einzelner Baum zeigt kaum einen Unterschied zwischen blatt- und ebenenweise:** der erste Versuch verglich EINEN Baum je Wachstumsart und fand fast identische Testfehler. Grund:
  ohne ein bindendes Blattbudget wachsen beide Varianten bis alle Splits mit positivem Gain gemacht sind – dieselbe MENGE an Splits, nur in anderer Reihenfolge, was am Ende dasselbe
  Ergebnis liefert (geprüft und bestätigt in `tests/test_algorithm.py`). Der Unterschied zeigt sich erst, wenn `num_leaves` klar VOR der natürlichen Sättigung greift, UND über ein ganzes
  Boosting-Ensemble gemittelt (ein einzelner Baum reicht nicht) – erst dann zeigt sich die erwartete Richtung (klein/verrauscht: ebenenweise leicht robuster; groß/sauber: blattweise gewinnt).
- **Der Histogramm-Vorteil ist bei kleinen Daten negativ, nicht nur "kleiner":** die erste Erwartung war "Histogramme sind immer schneller, nur der Faktor wächst mit n". Gemessen zeigt sich:
  bei sehr wenigen Trainingszeilen prüft die Histogramm-Suche MEHR Kandidaten als eine exakte Suche (die feste Bin-Zahl schlägt zu Buche, obwohl die Knoten kaum mehr Zeilen haben als Bins) –
  der Vorteil kippt erst bei rund 550–600 Zeilen ins Positive.
- **Kein `min_child_samples`-Bug übersehen:** ein erster Entwurf prüfte die Mindestblattgröße nur grob auf Knotenebene (Gesamtzahl vor dem Split), nicht je Kind aus dem tatsächlichen
  Zähl-Histogramm – ein Test mit `min_child_samples=20` deckte auf, dass einzelne Blätter mit nur 5 Zeilen entstanden. Behoben durch ein drittes Histogramm (Zeilenzahl je Bin) neben Gradient
  und Hesse, das den Split jetzt auch nach Zeilenzahl je Kind einschränkt.

## Verifikation

`tests/test_algorithm.py` (10 Tests): Histogramm-Split-Suche exakt gegen Brute-Force über dieselben Bin-Grenzen; Differenz-Trick exakt gegen direkte Neuberechnung; blattweise erzeugt bei
GLEICHER Blattzahl eine strukturell andere Baumform als ebenenweise (Tiefenbereich, gegen xgboost-demos exaktem Kern als Ebenenweise-Referenz); jeder gewählte Split hatte zum Zeitpunkt der
Wahl positiven Gain; Vorhersagen über Rang-/Fehlergrenzen gegen die echte `lightgbm`-Bibliothek; Grenzfälle (Mindest-Zeilenzahl je Blatt, γ prunt stärker, eine Runde, Reproduzierbarkeit der
Teilstichprobe). `tests/test_claims.py` (10 Tests) hält **jede Zahl** aus App und README fest. `tests/test_app.py` (21 Tests) prüft die Oberfläche per AppTest (jedes Preset, Aufgabenwechsel,
Abspielen mit rundenspezifischen Diagramm-Schlüsseln, Permalink, beide Experimente).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `lgbm_tree.py` | Baumkern: Bins, Histogramm, Differenz-Trick, blatt-/ebenenweises Wachsen (neu geschrieben) |
| `lgbm_algorithm.py` | Gradient/Hesse je Verlust, Fit, Vorhersage |
| `lgbm_scenario.py` | Lieferdaten (wortgleich aus xgboost-demo) |
| `lgbm_evaluation.py` | Analyse, Rundenkurve, Wachstumsart- und Zähler-Experimente |
| `lgbm_visualization.py` | Baum-, Karten-, Kurven- und Wichtigkeitsdiagramme |
| `lgbm_presets.py`, `lgbm_constants.py` | Regler, Permalink, Schnellstart-Beispiele, Grenzen |
| `tests/` | Algorithmus-, Claims- und App-Tests |

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\python -m pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -q
```

---

Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
