"""Modèles économétriques de la moyenne : baselines, ARIMA, SARIMA, SARIMAX.

Protocole sans fuite :
- Les paramètres sont estimés sur l'apprentissage seul.
- Ils sont ensuite gelés et appliqués à l'ensemble du jeu de données via
  ``res.apply(endog=y_full, exog=x_full)``, sans ré-estimation.
- Les prévisions à un pas obtenues via ``get_prediction(dynamic=False)``
  n'utilisent, pour chaque date t, que l'information disponible jusqu'à t−1.
"""

import time
import warnings
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "frozen.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Dataclass résultat
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Résultat complet d'un ajustement de modèle espace d'états."""
    nom: str
    order: tuple = (0, 0, 0)
    seasonal_order: tuple = (0, 0, 0, 0)
    aic: float = np.nan
    bic: float = np.nan
    params: pd.Series = field(default_factory=pd.Series)
    pvalues: pd.Series = field(default_factory=pd.Series)
    predictions: pd.Series = field(default_factory=pd.Series)
    residuals: pd.Series = field(default_factory=pd.Series)
    fit_seconds: float = np.nan
    predict_seconds: float = np.nan


# ---------------------------------------------------------------------------
# Préparation des données SARIMAX
# ---------------------------------------------------------------------------

def _build_exog(frame: pd.DataFrame) -> pd.Series:
    """Construit l'exogène : rendement SPY de la veille (spy_ret_{t-1}).

    Valeur connue à la clôture t−1 — aucune fuite temporelle.
    """
    return frame["spy_ret"].shift(1)


def _align_yx(y: pd.Series, x: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Aligne y et x sur les dates communes sans NaN."""
    valid = y.notna() & x.notna()
    return y[valid], x[valid]


# ---------------------------------------------------------------------------
# Baselines naïves
# ---------------------------------------------------------------------------

def naive_zero(frame: pd.DataFrame) -> FitResult:
    """Marche aléatoire : prévision constante r̂ = 0."""
    t0 = time.perf_counter()
    yhat = pd.Series(0.0, index=frame.index, name="naive_zero")
    t1 = time.perf_counter()
    return FitResult(
        nom="naive_zero",
        predictions=yhat,
        fit_seconds=0.0,
        predict_seconds=t1 - t0,
    )


def naive_last(frame: pd.DataFrame) -> FitResult:
    """Prévision = dernier rendement connu : r̂_t = r_{t-1}."""
    t0 = time.perf_counter()
    yhat = frame["dia_ret"].shift(1).rename("naive_last")
    t1 = time.perf_counter()
    return FitResult(
        nom="naive_last",
        predictions=yhat,
        fit_seconds=0.0,
        predict_seconds=t1 - t0,
    )


def historical_mean(frame: pd.DataFrame) -> FitResult:
    """Moyenne historique expansive décalée d'un pas.

    À la date t, prévision = moyenne de tous les rendements connus jusqu'à t−1.
    C'est le benchmark de référence pour le R²_oos de Campbell-Thompson.
    """
    t0 = time.perf_counter()
    yhat = (
        frame["dia_ret"]
        .expanding()
        .mean()
        .shift(1)
        .rename("historical_mean")
    )
    t1 = time.perf_counter()
    return FitResult(
        nom="historical_mean",
        predictions=yhat,
        fit_seconds=0.0,
        predict_seconds=t1 - t0,
    )


# ---------------------------------------------------------------------------
# Ajustement espace d'états (ARIMA / SARIMA / SARIMAX)
# ---------------------------------------------------------------------------

def fit_state_space(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    order: tuple = (1, 0, 1),
    seasonal_order: tuple = (0, 0, 0, 0),
    nom: str | None = None,
    with_exog: bool = False,
) -> FitResult:
    """Estime un modèle ARIMA/SARIMA/SARIMAX sur l'apprentissage, puis gèle les
    paramètres et applique le filtre Kalman sur la série complète.

    Protocole sans fuite :
    1. Estimation sur ``frame[train_mask]`` uniquement.
    2. ``res.apply(endog=y_full, exog=x_full)`` — aucun appel à ``refit``.
    3. ``get_prediction(dynamic=False)`` — prévisions à un pas, donc chaque
       prévision en t n'utilise que l'information disponible jusqu'à t−1.

    Paramètres
    ----------
    frame      : DataFrame complet (train + validation + test), colonnes dia_ret, spy_ret.
    train_mask : masque booléen, True sur la fenêtre d'apprentissage.
    order      : (p, d, q) ARIMA.
    seasonal_order : (P, D, Q, s) saisonnier.
    nom        : nom du modèle (généré automatiquement si None).
    with_exog  : si True, ajoute spy_ret_{t−1} comme variable exogène.
    """
    if nom is None:
        p, d, q = order
        P, D, Q, s = seasonal_order
        base = f"ARIMA({p},{d},{q})"
        if any([P, D, Q]):
            base = f"SARIMA{base[5:-1]}({P},{D},{Q})[{s}]"
        nom = base + ("X" if with_exog else "")

    # ---- Préparer y et x ------------------------------------------------
    x_full: pd.Series | None = None
    x_train: pd.Series | None = None

    if with_exog:
        x_full = _build_exog(frame)
        y_full, x_full = _align_yx(frame["dia_ret"], x_full)
        # Restreindre le masque d'apprentissage aux dates alignées
        train_idx = y_full.index[train_mask.reindex(y_full.index).fillna(False)]
        y_train = y_full.loc[train_idx]
        x_train = x_full.loc[train_idx]
    else:
        y_full = frame["dia_ret"]
        y_train = y_full[train_mask]

    # ---- Estimation sur l'apprentissage ---------------------------------
    t0_fit = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(
            endog=y_train,
            exog=x_train,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
        ).fit(disp=False)
    t1_fit = time.perf_counter()

    aic = float(res.aic)
    bic = float(res.bic)
    params = res.params.copy()
    pvalues = res.pvalues.copy()

    # Résidus d'apprentissage (les 10 premiers écartés car initialisation du filtre)
    residuals = pd.Series(res.resid.values, index=y_train.index).iloc[10:]

    # ---- Application sur la série complète (paramètres gelés) -----------
    t0_pred = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_full = res.apply(endog=y_full, exog=x_full)
        pred = res_full.get_prediction(dynamic=False).predicted_mean
    t1_pred = time.perf_counter()

    # Vérification interne : les paramètres sont bien identiques
    assert np.allclose(res.params.values, res_full.params.values, atol=1e-8), (
        "Paramètres modifiés après apply() — fuite possible !"
    )

    predictions = pd.Series(pred.values, index=y_full.index, name=nom)

    return FitResult(
        nom=nom,
        order=order,
        seasonal_order=seasonal_order,
        aic=aic,
        bic=bic,
        params=params,
        pvalues=pvalues,
        predictions=predictions,
        residuals=residuals,
        fit_seconds=t1_fit - t0_fit,
        predict_seconds=t1_pred - t0_pred,
    )


# ---------------------------------------------------------------------------
# Sélection automatique d'ordre ARIMA
# ---------------------------------------------------------------------------

def select_arima_order(
    frame: pd.DataFrame,
    train_mask: pd.Series,
) -> tuple[int, int, int]:
    """Grille ARIMA(p, 0, q) sur l'apprentissage, critère AIC.

    p, q ∈ [0, max_p] × [0, max_q] selon configs/frozen.yaml.
    d = 0 car les rendements log sont stationnaires (vérifiable via ADF).
    Retourne l'ordre (p, 0, q) qui minimise l'AIC.
    """
    cfg = _load_config()["econometrics"]
    max_p = cfg["max_p"]
    max_q = cfg["max_q"]

    y_train = frame["dia_ret"][train_mask]
    resultats = []

    for p, q in product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue  # modèle trivial
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(
                    endog=y_train,
                    order=(p, 0, q),
                    trend="c",
                ).fit(disp=False)
            resultats.append({"p": p, "q": q, "aic": res.aic, "bic": res.bic})
        except Exception:
            pass

    if not resultats:
        return (1, 0, 1)

    df_res = pd.DataFrame(resultats).sort_values("aic")
    best = df_res.iloc[0]
    return (int(best["p"]), 0, int(best["q"]))


def grille_arima(
    frame: pd.DataFrame,
    train_mask: pd.Series,
) -> pd.DataFrame:
    """Renvoie le tableau complet de la grille ARIMA trié par AIC."""
    cfg = _load_config()["econometrics"]
    max_p, max_q = cfg["max_p"], cfg["max_q"]
    y_train = frame["dia_ret"][train_mask]
    resultats = []

    for p, q in product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(endog=y_train, order=(p, 0, q), trend="c").fit(disp=False)
            resultats.append({
                "ordre": f"ARIMA({p},0,{q})",
                "p": p, "q": q,
                "AIC": round(res.aic, 2),
                "BIC": round(res.bic, 2),
            })
        except Exception:
            pass

    return pd.DataFrame(resultats).sort_values("AIC").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Test ADF
# ---------------------------------------------------------------------------

def adf_test(series: pd.Series, maxlag: int = 20) -> dict:
    """Test ADF (Augmented Dickey-Fuller) de stationnarité.

    H0 : présence d'une racine unitaire (non-stationnarité).
    Retourne un dictionnaire : statistique, p-value, lags, valeurs critiques.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, nlags, nobs, crit, _ = adfuller(series.dropna(), maxlag=maxlag)
    return {
        "stat": round(float(stat), 4),
        "pval": round(float(pval), 6),
        "nlags": nlags,
        "nobs": nobs,
        "critique_1pct":  round(crit["1%"],  4),
        "critique_5pct":  round(crit["5%"],  4),
        "critique_10pct": round(crit["10%"], 4),
        "stationnaire": pval < 0.05,
    }


# ---------------------------------------------------------------------------
# Diagnostics des résidus
# ---------------------------------------------------------------------------

def residual_diagnostics(residuals: pd.Series, lags: list[int] | None = None) -> dict:
    """Diagnostics complets sur les résidus d'apprentissage.

    Les 10 premiers résidus sont écartés (initialisation du filtre Kalman).

    Contrôles :
    - Ljung-Box sur résidus (H0 : pas d'autocorrélation)
    - Ljung-Box sur résidus² (H0 : pas d'effet ARCH)
    - Jarque-Bera (H0 : normalité)
    - Asymétrie et excès de kurtosis
    """
    if lags is None:
        lags = [5, 10, 20]

    r = residuals.iloc[10:].dropna()

    lb = acorr_ljungbox(r, lags=lags, return_df=True)
    lb2 = acorr_ljungbox(r ** 2, lags=lags, return_df=True)
    jb_stat, jb_pval, skew, kurt = jarque_bera(r)

    return {
        "ljung_box": {
            lag: {
                "stat": round(float(lb.loc[lag, "lb_stat"]), 4),
                "pval": round(float(lb.loc[lag, "lb_pvalue"]), 6),
            }
            for lag in lags
        },
        "ljung_box_squared": {
            lag: {
                "stat":  round(float(lb2.loc[lag, "lb_stat"]), 4),
                "pval":  round(float(lb2.loc[lag, "lb_pvalue"]), 6),
            }
            for lag in lags
        },
        "jarque_bera": {
            "stat": round(float(jb_stat), 4),
            "pval": round(float(jb_pval), 6),
            "normalite": jb_pval >= 0.05,
        },
        "asymetrie": round(float(skew), 4),
        "kurtosis_exces": round(float(kurt), 4),
        "n_residus": len(r),
    }


# ---------------------------------------------------------------------------
# Métriques de prévision
# ---------------------------------------------------------------------------

def metrics(
    y: pd.Series,
    yhat: pd.Series,
    bench: pd.Series,
) -> dict:
    """Métriques de prévision hors échantillon.

    Paramètres
    ----------
    y     : réalisations (rendements log DIA).
    yhat  : prévisions du modèle.
    bench : benchmark (moyenne historique expansive).

    Retourne
    --------
    rmse_pct  : RMSE en %.
    mae_pct   : MAE en %.
    dir_acc   : précision directionnelle (NaN si yhat constant = 0).
    r2_oos    : R²_oos de Campbell-Thompson vs benchmark.
    n         : nombre d'observations.
    """
    idx = y.index.intersection(yhat.index).intersection(bench.index)
    y, yhat, bench = y.loc[idx].dropna(), yhat.loc[idx].dropna(), bench.loc[idx].dropna()
    idx2 = y.index.intersection(yhat.index).intersection(bench.index)
    y, yhat, bench = y.loc[idx2], yhat.loc[idx2], bench.loc[idx2]
    n = len(y)

    if n == 0:
        return {"rmse_pct": np.nan, "mae_pct": np.nan,
                "dir_acc": np.nan, "r2_oos": np.nan, "n": 0}

    err = y - yhat
    rmse = float(np.sqrt((err ** 2).mean())) * 100
    mae  = float(err.abs().mean()) * 100

    # Précision directionnelle (NaN si prévisions toujours nulles)
    if yhat.abs().max() < 1e-12:
        dir_acc = np.nan
    else:
        dir_acc = float((np.sign(y) == np.sign(yhat)).mean())

    # R²_oos de Campbell-Thompson
    ss_model = float((err ** 2).sum())
    ss_bench = float(((y - bench) ** 2).sum())
    r2_oos = 1.0 - ss_model / ss_bench if ss_bench > 0 else np.nan

    return {
        "rmse_pct": round(rmse, 6),
        "mae_pct":  round(mae, 6),
        "dir_acc":  round(dir_acc, 4) if not np.isnan(dir_acc) else np.nan,
        "r2_oos":   round(r2_oos, 6),
        "n":        n,
    }


# ---------------------------------------------------------------------------
# Test de Diebold-Mariano
# ---------------------------------------------------------------------------

def diebold_mariano(
    y: pd.Series,
    f1: pd.Series,
    f2: pd.Series,
) -> dict:
    """Test de Diebold-Mariano (perte quadratique, bilatéral, h=1).

    H0 : les deux modèles ont la même précision (E[d_t] = 0).
    d_t = e1_t² − e2_t²

    Pour h=1, la variance de d_t est iid sous H0 — pas de HAC requis.
    Correction Harvey-Leybourne-Newbold appliquée (t avec n−1 degrés de liberté).

    Paramètres
    ----------
    y  : réalisations.
    f1 : prévisions du modèle 1.
    f2 : prévisions du modèle 2.

    Retourne
    --------
    dm_stat, pval, mean_d (>0 = f2 meilleur que f1).
    """
    idx = y.index.intersection(f1.index).intersection(f2.index)
    y, f1, f2 = y.loc[idx].dropna(), f1.loc[idx].dropna(), f2.loc[idx].dropna()
    idx2 = y.index.intersection(f1.index).intersection(f2.index)
    y, f1, f2 = y.loc[idx2], f1.loc[idx2], f2.loc[idx2]
    n = len(y)

    if n < 3:
        return {"dm_stat": np.nan, "pval": np.nan, "mean_d": np.nan, "n": n}

    e1 = (y - f1) ** 2
    e2 = (y - f2) ** 2
    d  = e1 - e2

    mean_d = float(d.mean())
    var_d  = float(d.var(ddof=1))

    if var_d < 1e-30:
        return {"dm_stat": np.nan, "pval": np.nan, "mean_d": mean_d, "n": n}

    # Correction HLN pour petits échantillons (h=1)
    h = 1
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat = float(mean_d / np.sqrt(var_d / n) * correction)

    pval = float(2 * (1 - scipy_stats.t.cdf(abs(dm_stat), df=n - 1)))

    return {
        "dm_stat": round(dm_stat, 4),
        "pval":    round(pval, 6),
        "mean_d":  round(mean_d, 8),
        "n":       n,
    }
