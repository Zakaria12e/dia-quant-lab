"""Génère et exécute notebooks/03_econometrie_moyenne.ipynb."""

import sys
import subprocess
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "03_econometrie_moyenne.ipynb"


# ─────────────────────────────────────────────────────────────────────────────
# Contenu des cellules
# ─────────────────────────────────────────────────────────────────────────────

CELL_TITRE = """\
# D03 — Économétrie de la moyenne : ARIMA · SARIMA · SARIMAX

**Projet** : Trading Intelligent — DIA / SPY
**Auteur** : Zakaria Elbidali
**Date** : 2026-11-19

> Toutes les estimations sont réalisées sur la **fenêtre d'apprentissage uniquement**.
> Les paramètres sont ensuite gelés et appliqués à la série complète via `res.apply()`.
> Aucune information hors-apprentissage ne contamine les estimations.
"""

CELL_IMPORTS = """\
import os
import sys
import warnings
from pathlib import Path

# --- Trouver la racine du projet (contient configs/frozen.yaml) ---
_candidate = Path.cwd()
for _ in range(6):
    if (_candidate / "configs" / "frozen.yaml").exists():
        break
    _candidate = _candidate.parent
else:
    raise RuntimeError("Racine du projet introuvable depuis " + str(Path.cwd()))
root = _candidate
os.chdir(str(root))
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
# -----------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.data import load_closes
from src.features import build_frame, split
from src.models.econometrics import (
    FitResult,
    fit_state_space,
    naive_zero,
    naive_last,
    historical_mean,
    select_arima_order,
    grille_arima,
    adf_test,
    residual_diagnostics,
    metrics,
    diebold_mariano,
)

REP_D03 = root / "reports" / "d03"
REP_FIG = REP_D03 / "figures"
REP_FIG.mkdir(parents=True, exist_ok=True)

print("Imports OK")
print("  Rapports :", REP_D03)
print("  Figures  :", REP_FIG)
"""

CELL_DONNEES = """\
closes = load_closes()
frame  = build_frame(closes)
dec    = split(frame)

train_mask = pd.Series(False, index=frame.index)
train_mask.loc[dec["train"].index] = True

print("Données chargées")
print("  Total séances   :", len(frame))
print("  Apprentissage   :", int(train_mask.sum()))
print("  Validation      :", len(dec["validation"]))
print("  Test            :", len(dec["test"]))
print("  Période totale  :", str(frame.index[0].date()), "→", str(frame.index[-1].date()))
"""

CELL_ADF_MD = """\
## 1. Test de stationnarité — ADF

**H₀** : présence d'une racine unitaire (série non stationnaire).
Rejet de H₀ à 5 % → rendements log stationnaires, **d = 0 justifié**.
Test effectué sur la fenêtre d'apprentissage uniquement.
"""

CELL_ADF = """\
adf_dia = adf_test(dec["train"]["dia_ret"])
adf_spy = adf_test(dec["train"]["spy_ret"])

df_adf = pd.DataFrame([
    {
        "Série": "DIA (apprentissage)",
        "Stat ADF": adf_dia["stat"],
        "p-value": adf_dia["pval"],
        "Critique 1%": adf_dia["critique_1pct"],
        "Critique 5%": adf_dia["critique_5pct"],
        "Stationnaire (p<0.05)": adf_dia["stationnaire"],
    },
    {
        "Série": "SPY (apprentissage)",
        "Stat ADF": adf_spy["stat"],
        "p-value": adf_spy["pval"],
        "Critique 1%": adf_spy["critique_1pct"],
        "Critique 5%": adf_spy["critique_5pct"],
        "Stationnaire (p<0.05)": adf_spy["stationnaire"],
    },
])
display(df_adf)
"""

CELL_ACF = """\
r_train = dec["train"]["dia_ret"].dropna()

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
plot_acf(r_train,     lags=25, ax=axes[0, 0],
         title="ACF — Rendements DIA (apprentissage)")
plot_pacf(r_train,    lags=25, ax=axes[0, 1],
          title="PACF — Rendements DIA (apprentissage)")
plot_acf(r_train**2,  lags=25, ax=axes[1, 0],
         title="ACF — Rendements² DIA (effets ARCH)")
plot_pacf(r_train**2, lags=25, ax=axes[1, 1],
          title="PACF — Rendements² DIA")

for ax in axes.flat:
    ax.set_xlabel("Lag")
plt.tight_layout()
fig.savefig(REP_FIG / "acf_pacf_dia.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Figure sauvegardée :", str(REP_FIG / "acf_pacf_dia.png"))
"""

CELL_GRILLE_MD = """\
## 2. Grille ARIMA(p, 0, q) — sélection par AIC

d = 0 : les rendements log sont stationnaires (voir ADF).
Grille : p ∈ [0, 3], q ∈ [0, 3] selon `configs/frozen.yaml`.
Estimation sur l'apprentissage uniquement — l'AIC sert de critère de sélection.
"""

CELL_GRILLE = """\
print("Calcul de la grille ARIMA — peut prendre ~1 min…")
df_grille = grille_arima(frame, train_mask)

p_best = int(df_grille.iloc[0]["p"])
q_best = int(df_grille.iloc[0]["q"])
meilleur_ordre = (p_best, 0, q_best)

print("Meilleur ordre AIC :", meilleur_ordre)
print()
print("Top 8 par AIC :")
display(df_grille.head(8))
"""

CELL_ARIMA_FIT = """\
print("Estimation ARIMA" + str(meilleur_ordre) + " sur l'apprentissage…")
res_arima = fit_state_space(
    frame, train_mask,
    order=meilleur_ordre,
    seasonal_order=(0, 0, 0, 0),
    with_exog=False,
)
print("AIC = " + str(round(res_arima.aic, 2)) + "  |  BIC = " + str(round(res_arima.bic, 2)))
print("Temps estimation : " + str(round(res_arima.fit_seconds, 2)) + " s")
print()
df_params = pd.DataFrame({
    "Paramètre": res_arima.params.index,
    "Valeur": res_arima.params.values.round(6),
    "p-value": res_arima.pvalues.values.round(4),
    "Significatif (5%)": (res_arima.pvalues.values < 0.05),
})
display(df_params)
"""

CELL_SARIMA_MD = """\
## 3. Extensions saisonnières — SARIMA(p,0,q)(P,0,Q)[5]

Période saisonnière : **s = 5** (cycle hebdomadaire, 5 séances de bourse).
Ordre non-saisonnier fixé au meilleur ordre ARIMA trouvé par la grille.
Trois variantes testées : (P, D, Q) = (1,0,0), (0,0,1), (1,0,1).
"""

CELL_SARIMA = """\
sarima_configs = [(1, 0, 0, 5), (0, 0, 1, 5), (1, 0, 1, 5)]
resultats_sarima = {}

for P, D, Q, s in sarima_configs:
    nom = (
        "SARIMA("
        + str(p_best) + ",0," + str(q_best)
        + ")(" + str(P) + "," + str(D) + "," + str(Q)
        + ")[" + str(s) + "]"
    )
    print("Estimation", nom, "…")
    try:
        res = fit_state_space(
            frame, train_mask,
            order=meilleur_ordre,
            seasonal_order=(P, D, Q, s),
            with_exog=False,
            nom=nom,
        )
        resultats_sarima[nom] = res
        print(
            "  AIC = " + str(round(res.aic, 2))
            + "  |  BIC = " + str(round(res.bic, 2))
            + "  |  t_fit = " + str(round(res.fit_seconds, 2)) + " s"
        )
    except Exception as exc:
        print("  ECHEC :", exc)
        resultats_sarima[nom] = None
"""

CELL_SARIMAX_MD = """\
## 4. SARIMAX — variable exogène SPY retard 1

**Exogène** : spy_ret_{t−1} (rendement SPY de la veille, connu à la clôture t).
Aucune information future n'est utilisée.
Ordre non-saisonnier : meilleur ordre ARIMA. Pas de composante saisonnière.
"""

CELL_SARIMAX = """\
print("Estimation SARIMAX…")
res_sarimax = fit_state_space(
    frame, train_mask,
    order=meilleur_ordre,
    seasonal_order=(0, 0, 0, 0),
    with_exog=True,
)
print("AIC = " + str(round(res_sarimax.aic, 2)) + "  |  BIC = " + str(round(res_sarimax.bic, 2)))
print("Temps estimation : " + str(round(res_sarimax.fit_seconds, 2)) + " s")
print()
df_sarimax_params = pd.DataFrame({
    "Paramètre": res_sarimax.params.index,
    "Valeur": res_sarimax.params.values.round(6),
    "p-value": res_sarimax.pvalues.values.round(4),
    "Significatif (5%)": (res_sarimax.pvalues.values < 0.05),
})
display(df_sarimax_params)
"""

CELL_DIAG_MD = """\
## 5. Diagnostics des résidus

Modèle de référence : meilleur ARIMA.
Tests appliqués aux résidus d'apprentissage (premiers 10 écartés — initialisation du filtre Kalman) :
- **Ljung-Box** (H₀ : pas d'autocorrélation)
- **Ljung-Box sur r²** (H₀ : pas d'effet ARCH)
- **Jarque-Bera** (H₀ : normalité)
"""

CELL_DIAG = """\
diag = residual_diagnostics(res_arima.residuals)

rows_lb = []
for lag, v in diag["ljung_box"].items():
    rows_lb.append({
        "Lag": lag,
        "LB stat (résidus)": v["stat"],
        "p-value (résidus)": v["pval"],
        "Rejette H0 (5%)": v["pval"] < 0.05,
        "LB stat (résidus²)": diag["ljung_box_squared"][lag]["stat"],
        "p-value (résidus²)": diag["ljung_box_squared"][lag]["pval"],
        "ARCH détecté (5%)": diag["ljung_box_squared"][lag]["pval"] < 0.05,
    })
df_lb = pd.DataFrame(rows_lb)
display(df_lb)

print()
jb = diag["jarque_bera"]
print("Jarque-Bera : stat = " + str(jb["stat"]) + "  |  p-value = " + str(jb["pval"]))
print("Normalité (p>=0.05) :", jb["normalite"])
print("Asymétrie :", diag["asymetrie"])
print("Excès de kurtosis :", diag["kurtosis_exces"])
print("Nombre de résidus utilisés :", diag["n_residus"])

r_resid = res_arima.residuals.dropna()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(r_resid,     lags=25, ax=axes[0], title="ACF résidus ARIMA")
plot_acf(r_resid**2,  lags=25, ax=axes[1], title="ACF résidus² ARIMA (test ARCH)")
plt.tight_layout()
fig.savefig(REP_FIG / "acf_residus_arima.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print()
print("Figure sauvegardée :", str(REP_FIG / "acf_residus_arima.png"))

rows_csv = []
for lag, v in diag["ljung_box"].items():
    rows_csv.append({
        "modele": res_arima.nom, "test": "ljung_box_residus",
        "lag": lag, "stat": v["stat"], "pval": v["pval"],
    })
for lag, v in diag["ljung_box_squared"].items():
    rows_csv.append({
        "modele": res_arima.nom, "test": "ljung_box_residus_carres",
        "lag": lag, "stat": v["stat"], "pval": v["pval"],
    })
rows_csv.append({
    "modele": res_arima.nom, "test": "jarque_bera",
    "lag": None, "stat": jb["stat"], "pval": jb["pval"],
})
pd.DataFrame(rows_csv).to_csv(REP_D03 / "diagnostics_residus.csv", index=False)
print("CSV sauvegardé :", str(REP_D03 / "diagnostics_residus.csv"))
"""

CELL_VAL_MD = """\
## 6. Comparatif — fenêtre de validation (2023-01-03 → 2024-12-31)

Toutes les prévisions utilisent des **paramètres gelés** estimés sur l'apprentissage.
Référence Campbell-Thompson : `historical_mean` (moyenne historique expansive décalée d'un pas).
Le meilleur modèle est sélectionné ici pour être évalué sur le test.
"""

CELL_VAL = """\
res_zero = naive_zero(frame)
res_last = naive_last(frame)
res_hm   = historical_mean(frame)

val_idx   = dec["validation"].index
y_val     = frame.loc[val_idx, "dia_ret"]
bench_val = res_hm.predictions.reindex(val_idx)

tous_modeles = {
    "naive_zero":      res_zero,
    "naive_last":      res_last,
    "historical_mean": res_hm,
    res_arima.nom:     res_arima,
    res_sarimax.nom:   res_sarimax,
}
for nom, res in resultats_sarima.items():
    if res is not None:
        tous_modeles[nom] = res

rows_val = []
for nom, res in tous_modeles.items():
    yhat = res.predictions.reindex(val_idx)
    m = metrics(y_val, yhat, bench_val)
    rows_val.append({
        "modele":    nom,
        "RMSE (%)":  m["rmse_pct"],
        "MAE (%)":   m["mae_pct"],
        "Dir acc":   m["dir_acc"],
        "R²_oos":    m["r2_oos"],
        "n":         m["n"],
    })

df_val = pd.DataFrame(rows_val).sort_values("RMSE (%)").reset_index(drop=True)
display(df_val)
df_val.to_csv(REP_D03 / "comparatif_validation.csv", index=False)

meilleur_val_nom = df_val.iloc[0]["modele"]
print()
print("CSV sauvegardé :", str(REP_D03 / "comparatif_validation.csv"))
print("Meilleur modèle (validation, RMSE) :", meilleur_val_nom)
"""

CELL_TEST_MD = """\
## 7. Évaluation hors échantillon — fenêtre de test (2025-01-02 → 2026-10-30)

Tous les modèles sont évalués sur le test avec les **mêmes paramètres gelés**.
Test de Diebold-Mariano (correction Harvey-Leybourne-Newbold) :
H₀ = égale précision vs `historical_mean` (benchmark Campbell-Thompson).
La sélection du meilleur modèle était faite **avant** de regarder le test.
"""

CELL_TEST = """\
test_idx   = dec["test"].index
y_test     = frame.loc[test_idx, "dia_ret"]
bench_test = res_hm.predictions.reindex(test_idx)

rows_test = []
for nom, res in tous_modeles.items():
    yhat = res.predictions.reindex(test_idx)
    m    = metrics(y_test, yhat, bench_test)
    dm   = diebold_mariano(y_test, res_hm.predictions.reindex(test_idx), yhat)
    rows_test.append({
        "modele":      nom,
        "RMSE (%)":    m["rmse_pct"],
        "MAE (%)":     m["mae_pct"],
        "Dir acc":     m["dir_acc"],
        "R²_oos":      m["r2_oos"],
        "DM stat":     dm["dm_stat"],
        "DM p-value":  dm["pval"],
        "n":           m["n"],
    })

df_test = pd.DataFrame(rows_test).sort_values("RMSE (%)").reset_index(drop=True)
display(df_test)
df_test.to_csv(REP_D03 / "comparatif_test.csv", index=False)
print("CSV sauvegardé :", str(REP_D03 / "comparatif_test.csv"))

meilleur_test_nom = df_test.iloc[0]["modele"]
res_meilleur = tous_modeles[meilleur_test_nom]
yhat_best = res_meilleur.predictions.reindex(test_idx)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y_test.index, y_test.values * 100,
        label="Réalisé DIA", color="#1f77b4", alpha=0.65, lw=1)
ax.plot(yhat_best.index, yhat_best.values * 100,
        label="Prévu — " + meilleur_test_nom,
        color="#ff7f0e", alpha=0.85, lw=1.1)
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_xlabel("Date")
ax.set_ylabel("Rendement log (%)")
ax.set_title("Prévision vs Réalisé — Fenêtre de test")
ax.legend()
plt.tight_layout()
fig.savefig(REP_FIG / "prevision_vs_realise_test.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Figure sauvegardée :", str(REP_FIG / "prevision_vs_realise_test.png"))
"""

CELL_PREVISIONS = """\
df_previsions = pd.DataFrame({"dia_ret_realise": frame["dia_ret"]})
for nom, res in tous_modeles.items():
    df_previsions[nom] = res.predictions.reindex(frame.index)
df_previsions.index.name = "date"
df_previsions.to_csv(REP_D03 / "previsions_datees.csv")
print("CSV sauvegardé :", str(REP_D03 / "previsions_datees.csv"))
print("Forme :", df_previsions.shape)

rows_temps = []
for nom, res in tous_modeles.items():
    rows_temps.append({
        "modele":           nom,
        "fit_secondes":     round(res.fit_seconds, 4),
        "predict_secondes": round(res.predict_seconds, 4),
        "total_secondes":   round(res.fit_seconds + res.predict_seconds, 4),
    })
df_temps = (
    pd.DataFrame(rows_temps)
    .sort_values("total_secondes", ascending=False)
    .reset_index(drop=True)
)
display(df_temps)
df_temps.to_csv(REP_D03 / "temps_calcul.csv", index=False)
print("CSV sauvegardé :", str(REP_D03 / "temps_calcul.csv"))
"""

CELL_CONCL_MD = """\
## 8. Conclusion

Conclusions établies à partir des résultats réellement obtenus — aucun chiffre prédéfini.
"""

CELL_CONCL = """\
sep = "=" * 62
print(sep)
print("CONCLUSIONS — D03 ÉCONOMÉTRIE DE LA MOYENNE")
print(sep)
print()

conc_dia = "stationnaire" if adf_dia["stationnaire"] else "non stationnaire"
conc_spy = "stationnaire" if adf_spy["stationnaire"] else "non stationnaire"
print("1. STATIONNARITÉ")
print("   DIA : " + conc_dia
      + "  (ADF=" + str(adf_dia["stat"])
      + ", p=" + str(adf_dia["pval"]) + ")")
print("   SPY : " + conc_spy
      + "  (ADF=" + str(adf_spy["stat"])
      + ", p=" + str(adf_spy["pval"]) + ")")
print()

print("2. SÉLECTION D'ORDRE ARIMA")
print("   Meilleur ordre : " + str(meilleur_ordre)
      + "  (AIC=" + str(round(res_arima.aic, 2)) + ")")
print()

best_val_row  = df_val.iloc[0]
print("3. COMPARATIF VALIDATION — Meilleur modèle : " + str(meilleur_val_nom))
print("   RMSE = " + str(best_val_row["RMSE (%)"]) + " %"
      + "  |  Dir acc = " + str(best_val_row["Dir acc"])
      + "  |  R²_oos = " + str(best_val_row["R²_oos"]))
print()

best_test_row = df_test.iloc[0]
print("4. ÉVALUATION TEST — Meilleur modèle : " + str(meilleur_test_nom))
print("   RMSE = " + str(best_test_row["RMSE (%)"]) + " %"
      + "  |  Dir acc = " + str(best_test_row["Dir acc"])
      + "  |  R²_oos = " + str(best_test_row["R²_oos"]))
print()

lb10  = diag["ljung_box"][10]
lb2_10 = diag["ljung_box_squared"][10]
conc_lb = (
    "autocorrélation résiduelle détectée"
    if lb10["pval"] < 0.05
    else "pas d'autocorrélation résiduelle"
)
conc_arch = (
    "effets ARCH détectés"
    if lb2_10["pval"] < 0.05
    else "pas d'effets ARCH significatifs"
)
print("5. DIAGNOSTICS RÉSIDUS (meilleur ARIMA)")
print("   " + conc_lb
      + "  (LB lag=10, p=" + str(lb10["pval"]) + ")")
print("   " + conc_arch
      + "  (LB² lag=10, p=" + str(lb2_10["pval"]) + ")")
print("   Jarque-Bera p=" + str(jb["pval"])
      + "  — normalité : " + str(jb["normalite"]))
print()

sarimax_val_row = df_val[df_val["modele"] == res_sarimax.nom]
sarimax_test_row = df_test[df_test["modele"] == res_sarimax.nom]
if len(sarimax_val_row) and len(sarimax_test_row):
    sarimax_rmse_val  = round(float(sarimax_val_row.iloc[0]["RMSE (%)"]), 4)
    arima_rmse_val    = round(float(df_val[df_val["modele"] == res_arima.nom].iloc[0]["RMSE (%)"]), 4)
    sarimax_rmse_test = round(float(sarimax_test_row.iloc[0]["RMSE (%)"]), 4)
    arima_rmse_test   = round(float(df_test[df_test["modele"] == res_arima.nom].iloc[0]["RMSE (%)"]), 4)
    print("6. APPORT DE L'EXOGÈNE SPY")
    if sarimax_rmse_val < arima_rmse_val:
        print("   SARIMAX améliore ARIMA en validation"
              + " (RMSE " + str(sarimax_rmse_val)
              + " < " + str(arima_rmse_val) + ")")
    else:
        print("   SARIMAX n'améliore pas ARIMA en validation"
              + " (RMSE " + str(sarimax_rmse_val)
              + " vs " + str(arima_rmse_val) + ")")
    if sarimax_rmse_test < arima_rmse_test:
        print("   SARIMAX améliore ARIMA en test"
              + " (RMSE " + str(sarimax_rmse_test)
              + " < " + str(arima_rmse_test) + ")")
    else:
        print("   SARIMAX n'améliore pas ARIMA en test"
              + " (RMSE " + str(sarimax_rmse_test)
              + " vs " + str(arima_rmse_test) + ")")
print()
print(sep)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Construction du notebook
# ─────────────────────────────────────────────────────────────────────────────

cells = [
    new_markdown_cell(CELL_TITRE),
    new_code_cell(CELL_IMPORTS),
    new_code_cell(CELL_DONNEES),
    new_markdown_cell(CELL_ADF_MD),
    new_code_cell(CELL_ADF),
    new_code_cell(CELL_ACF),
    new_markdown_cell(CELL_GRILLE_MD),
    new_code_cell(CELL_GRILLE),
    new_code_cell(CELL_ARIMA_FIT),
    new_markdown_cell(CELL_SARIMA_MD),
    new_code_cell(CELL_SARIMA),
    new_markdown_cell(CELL_SARIMAX_MD),
    new_code_cell(CELL_SARIMAX),
    new_markdown_cell(CELL_DIAG_MD),
    new_code_cell(CELL_DIAG),
    new_markdown_cell(CELL_VAL_MD),
    new_code_cell(CELL_VAL),
    new_markdown_cell(CELL_TEST_MD),
    new_code_cell(CELL_TEST),
    new_code_cell(CELL_PREVISIONS),
    new_markdown_cell(CELL_CONCL_MD),
    new_code_cell(CELL_CONCL),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.14.0",
}

with open(NB_PATH, "w", encoding="utf-8") as fh:
    nbformat.write(nb, fh)

print("Notebook écrit :", NB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Exécution via nbconvert
# ─────────────────────────────────────────────────────────────────────────────

print("Exécution du notebook (peut prendre ~3 min)…")
result = subprocess.run(
    [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        "--ExecutePreprocessor.kernel_name=python3",
        str(NB_PATH),
    ],
    cwd=str(ROOT),
    capture_output=True,
    text=True,
)

print(result.stdout)
if result.returncode != 0:
    print("ERREUR nbconvert :")
    print(result.stderr[-3000:])
    sys.exit(1)
else:
    print("Notebook exécuté avec succès.")
    if result.stderr:
        print("Avertissements :")
        print(result.stderr[-1000:])
