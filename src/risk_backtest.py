"""Vérification hors échantillon des mesures de risque (D04).

Toutes les fonctions comparent une mesure PRÉVUE (datée par sa date cible) à
l'observation FUTURE correspondante. Elles renvoient des dict ou des
DataFrames, sans print ni graphique, et sont réutilisables par le backtest
(D06), l'application (D07) et le paper trading (D08).

Conventions d'unités (identiques à ``src/risk.py``) :
- rendements r en % par jour (100 × rendement log) ;
- VaR et ES en PERTE POSITIVE, % par jour ; violation le jour t ⇔ r_t < −VaR_t ;
- variances (réalisée et prévue) en %² par jour.

Tests implémentés
-----------------
- Kupiec (1995) : couverture inconditionnelle, LR_uc ~ χ²(1).
- Christoffersen (1998) : indépendance des violations, LR_ind ~ χ²(1) ;
  couverture conditionnelle LR_cc = LR_uc + LR_ind ~ χ²(2).
- Acerbi & Székely (2014) : statistique Z2 de l'ES, p-value par simulation.
- Patton (2011) : pertes QLIKE et MSE pour la volatilité prévue.
- Diebold-Mariano sur les pertes (variance de Newey-West, correction HLN).
- Mincer-Zarnowitz : régression réalisé = a + b·prévu, test joint a = 0, b = 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats
from scipy.special import xlogy

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "frozen.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _seuil() -> float:
    return float(_load_config()["risk"]["seuil_tests"])


def _aligner(*series: pd.Series) -> list[pd.Series]:
    """Aligne plusieurs Series sur leurs dates communes sans NaN."""
    df = pd.concat([pd.Series(s) for s in series], axis=1).dropna()
    return [df.iloc[:, i] for i in range(df.shape[1])]


def _nw_lags(n: int) -> int:
    """Nombre de retards de Newey-West : ⌊4 (n/100)^(2/9)⌋ (règle de Newey-West, 1994)."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


# ---------------------------------------------------------------------------
# Violations de VaR
# ---------------------------------------------------------------------------

def violations(r_pct: pd.Series, var_pct: pd.Series) -> pd.Series:
    """Indicatrice de violation : 1 le jour t si r_t < −VaR_t (perte > VaR).

    r_pct en % par jour, var_pct en perte positive (% par jour), alignés par
    date cible. Les dates sans VaR ou sans rendement sont écartées.
    """
    r, v = _aligner(r_pct, var_pct)
    return (r < -v).astype(int).rename("violation")


# ---------------------------------------------------------------------------
# Kupiec et Christoffersen
# ---------------------------------------------------------------------------

def kupiec_test(hits, alpha: float) -> dict:
    """Test de couverture inconditionnelle de Kupiec (1995).

    H0 : la probabilité de violation vaut alpha.
    LR_uc = −2 ln[(1−α)^(n−x) α^x] + 2 ln[(1−π̂)^(n−x) π̂^x],  π̂ = x/n,  ~ χ²(1).

    Paramètres
    ----------
    hits  : suite de 0/1 (1 = violation), sans unité.
    alpha : taux de violation attendu (0.05 pour la VaR 95 %).

    Retourne un dict : n, violations, taux_observe, taux_attendu, lr_uc, pval.
    """
    h = np.asarray(hits, dtype=int)
    n, x = int(h.size), int(h.sum())
    if n == 0:
        return {"n": 0, "violations": 0, "taux_observe": np.nan,
                "taux_attendu": alpha, "lr_uc": np.nan, "pval": np.nan}
    pi = x / n
    ll0 = xlogy(n - x, 1 - alpha) + xlogy(x, alpha)
    ll1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return {"n": n, "violations": x, "taux_observe": pi, "taux_attendu": alpha,
            "lr_uc": float(lr), "pval": float(stats.chi2.sf(lr, df=1))}


def christoffersen_test(hits, alpha: float | None = None) -> dict:
    """Test d'indépendance des violations de Christoffersen (1998), et couverture conditionnelle.

    Chaîne de Markov d'ordre 1 sur les violations : n_ij = nombre de passages
    de l'état i (veille) à l'état j (jour). π01 = n01/(n00+n01), π11 = n11/(n10+n11).
    H0 (indépendance) : π01 = π11.
    LR_ind = −2 [ln L(π̂) − ln L(π̂01, π̂11)] ~ χ²(1).
    Si ``alpha`` est fourni : LR_cc = LR_uc + LR_ind ~ χ²(2) (couverture
    conditionnelle : bon taux ET violations indépendantes), LR_uc de Kupiec
    calculé sur la même suite.

    Retourne un dict : n00, n01, n10, n11, pi01, pi11, lr_ind, pval_ind,
    et si alpha : lr_uc, lr_cc, pval_cc.
    """
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if h.size > 1 else 0.0
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
           + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    lr_ind = max(-2.0 * (ll0 - ll1), 0.0)
    out = {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
           "pi01": pi01, "pi11": pi11,
           "lr_ind": float(lr_ind), "pval_ind": float(stats.chi2.sf(lr_ind, df=1))}
    if alpha is not None:
        lr_uc = kupiec_test(h, alpha)["lr_uc"]
        lr_cc = lr_uc + lr_ind
        out.update({"lr_uc": float(lr_uc), "lr_cc": float(lr_cc),
                    "pval_cc": float(stats.chi2.sf(lr_cc, df=2))})
    return out


# ---------------------------------------------------------------------------
# Test de l'Expected Shortfall (Acerbi & Székely, 2014)
# ---------------------------------------------------------------------------

def z2_statistic(r, var, es, alpha: float) -> np.ndarray | float:
    """Statistique Z2 d'Acerbi & Székely (2014), sans unité.

    Z2 = Σ_t r_t · I_t / (T · α · ES_t) + 1,  I_t = 1{r_t < −VaR_t},
    avec r en % et VaR/ES en perte positive (%). Sous H0, E[Z2] = 0 ;
    Z2 < 0 : l'ES sous-estime les pertes (trop optimiste) ;
    Z2 > 0 : l'ES est prudent.
    Accepte r de forme (T,) ou (n_sim, T) ; var et es de forme (T,).
    """
    r = np.asarray(r, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    T = var.size
    hits = r < -var
    return np.sum(np.where(hits, r / es, 0.0), axis=-1) / (T * alpha) + 1.0


def simulateur_parametrique(mu, sigma, dist_obj, dist_params=None) -> Callable:
    """Simulateur de rendements sous la loi d'un modèle paramétrique.

    r*_t = μ_t + σ_t z*_t, z* tiré de la loi STANDARDISÉE ``dist_obj``
    (normal, t, skewt d'arch) par inversion : z* = F⁻¹(U), U ~ U(0, 1).
    μ et σ en % par jour, alignés sur les dates testées.

    Retourne une fonction (rng, n_sim) → ndarray (n_sim, T) en % par jour.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    def simuler(rng: np.random.Generator, n_sim: int) -> np.ndarray:
        u = rng.uniform(size=(n_sim, sigma.size))
        u = np.clip(u, 1e-12, 1 - 1e-12)
        z = np.asarray(dist_obj.ppf(u.ravel(), dist_params)).reshape(u.shape)
        return mu + sigma * z

    return simuler


def simulateur_historique(r_pct_full: pd.Series, dates_cibles, window: int) -> Callable:
    """Simulateur sous la loi de la simulation historique.

    Pour chaque date cible t, r*_t est tiré uniformément parmi les ``window``
    rendements r_{t−window}, …, r_{t−1} (ceux qui ont servi à la VaR HS).

    Retourne une fonction (rng, n_sim) → ndarray (n_sim, T) en % par jour.
    """
    r = pd.Series(r_pct_full).dropna()
    valeurs = r.to_numpy()
    pos = r.index.get_indexer(pd.DatetimeIndex(dates_cibles))
    if (pos < window).any():
        raise ValueError("Pas assez d'historique avant certaines dates cibles.")

    def simuler(rng: np.random.Generator, n_sim: int) -> np.ndarray:
        k = rng.integers(1, window + 1, size=(n_sim, pos.size))
        return valeurs[pos[None, :] - k]

    return simuler


def es_backtest_acerbi_szekely(r, var, es, alpha: float,
                               simulateur: Callable | None = None,
                               n_sim: int | None = None,
                               seed: int | None = None) -> dict:
    """Backtest de l'ES par la statistique Z2 d'Acerbi & Székely (2014).

    p-value unilatérale par simulation sous la loi du modèle :
    p = P(Z2* ≤ Z2_obs), avec Z2* calculé sur des rendements simulés par
    ``simulateur`` (``simulateur_parametrique`` ou ``simulateur_historique``)
    et les MÊMES VaR/ES prévues. Une petite p-value signale un ES trop optimiste.
    n_sim et seed par défaut : ``risk.es_n_sim`` et ``seed`` de frozen.yaml.

    Paramètres
    ----------
    r, var, es : Series alignées par date cible (r en %, VaR/ES en perte positive %).
    alpha      : probabilité de queue (0.05 ou 0.01).

    Retourne un dict : z2, pval (NaN sans simulateur), n, violations,
    es_moyen_prevu_pct et perte_moyenne_realisee_pct (moyennes sur les jours
    de violation, en % par jour), ecart_es_pct (= perte réalisée − ES prévu ;
    > 0 : ES optimiste), n_sim.
    """
    cfg = _load_config()
    n_sim = int(n_sim if n_sim is not None else cfg["risk"]["es_n_sim"])
    seed = int(seed if seed is not None else cfg["seed"])
    r, var, es = _aligner(r, var, es)
    z2 = float(z2_statistic(r.to_numpy(), var.to_numpy(), es.to_numpy(), alpha))
    hits = (r < -var).to_numpy()

    pval = np.nan
    if simulateur is not None:
        rng = np.random.default_rng(seed)
        r_sim = simulateur(rng, n_sim)
        if r_sim.shape != (n_sim, len(r)):
            raise ValueError(
                f"Simulateur de forme {r_sim.shape}, attendu {(n_sim, len(r))} : il doit être "
                "construit sur exactement les dates cibles testées (sans NaN)."
            )
        z2_sim = z2_statistic(r_sim, var.to_numpy(), es.to_numpy(), alpha)
        pval = float(np.mean(z2_sim <= z2))

    es_moy = float(es[hits].mean()) if hits.any() else np.nan
    perte_moy = float(-r[hits].mean()) if hits.any() else np.nan
    return {
        "z2": z2, "pval": pval, "n": int(len(r)), "violations": int(hits.sum()),
        "es_moyen_prevu_pct": es_moy, "perte_moyenne_realisee_pct": perte_moy,
        "ecart_es_pct": perte_moy - es_moy if hits.any() else np.nan,
        "n_sim": n_sim if simulateur is not None else 0,
    }


def verifier_var_es(r, var, es, alpha: float, simulateur: Callable | None = None,
                    n_sim: int | None = None, seed: int | None = None) -> dict:
    """Ligne complète du tableau de vérification VaR/ES pour un modèle, une fenêtre, un niveau.

    Regroupe Kupiec, Christoffersen (indépendance, couverture conditionnelle)
    et Acerbi-Székely, puis formule une conclusion au seuil ``risk.seuil_tests``.

    Retourne un dict : niveau (%), jours, violations, taux_observe_pct,
    taux_attendu_pct, kupiec_p, christoffersen_ind_p, couverture_cond_p,
    es_moyen_prevu_pct, perte_moyenne_realisee_pct, z2, z2_p, conclusion.
    """
    seuil = _seuil()
    r, var, es = _aligner(r, var, es)
    hits = (r < -var).astype(int).to_numpy()
    k = kupiec_test(hits, alpha)
    c = christoffersen_test(hits, alpha)
    a = es_backtest_acerbi_szekely(r, var, es, alpha, simulateur, n_sim, seed)

    problemes = []
    if k["pval"] < seuil:
        problemes.append("taux de violation rejeté (Kupiec)")
    if c["pval_ind"] < seuil:
        problemes.append("violations groupées (Christoffersen)")
    if c["pval_cc"] < seuil:
        problemes.append("couverture conditionnelle rejetée")
    if not np.isnan(a["pval"]) and a["pval"] < seuil:
        problemes.append("ES trop optimiste (Z2)")
    conclusion = "validée" if not problemes else "rejetée : " + " ; ".join(problemes)
    if np.isnan(a["pval"]):
        conclusion += " ; ES non testé (Z2 sans simulateur)"

    return {
        "niveau": int(round(100 * (1 - alpha))),
        "jours": k["n"], "violations": k["violations"],
        "taux_observe_pct": 100 * k["taux_observe"],
        "taux_attendu_pct": 100 * alpha,
        "kupiec_p": k["pval"],
        "christoffersen_ind_p": c["pval_ind"],
        "couverture_cond_p": c["pval_cc"],
        "es_moyen_prevu_pct": a["es_moyen_prevu_pct"],
        "perte_moyenne_realisee_pct": a["perte_moyenne_realisee_pct"],
        "z2": a["z2"], "z2_p": a["pval"],
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# Qualité de la volatilité prévue
# ---------------------------------------------------------------------------

def vol_loss(realized_var: pd.Series, forecast_var: pd.Series) -> pd.DataFrame:
    """Pertes journalières de la variance prévue (Patton, 2011).

    QLIKE_t = RV_t / h_t + ln h_t  (forme de Patton, 2011, robuste au bruit de
    RV et définie même si RV_t = 0, par exemple r_t = 0). Elle ne diffère de
    RV/h − ln(RV/h) − 1 que par un terme qui ne dépend pas de la prévision :
    même classement, mêmes différences entre modèles (donc même test DM).
    Pour une RV donnée, elle est minimale en h_t = RV_t. Plus petit = meilleur.
    MSE_t = (RV_t − h_t)².

    Paramètres
    ----------
    realized_var : variance réalisée en %² par jour (r² ou Parkinson²).
    forecast_var : variance prévue σ² en %² par jour, même date cible.

    Retourne un DataFrame indexé par date : qlike (sans unité, variances en %²), mse (%⁴).
    """
    rv, h = _aligner(realized_var, forecast_var)
    return pd.DataFrame({"qlike": rv / h + np.log(h), "mse": (rv - h) ** 2}, index=rv.index)


def dm_test_loss(loss_a: pd.Series, loss_b: pd.Series) -> dict:
    """Test de Diebold-Mariano sur deux séries de pertes (par exemple QLIKE).

    d_t = L_a,t − L_b,t ; H0 : E[d_t] = 0 (précision égale).
    Les pertes de volatilité sont autocorrélées : la variance de d̄ est
    estimée par Newey-West (noyau de Bartlett, ⌊4(n/100)^(2/9)⌋ retards),
    avec correction de Harvey-Leybourne-Newbold et loi t(n−1). Bilatéral.

    Retourne un dict : dm_stat, pval, mean_d (> 0 : b meilleur que a), n, lags.
    """
    a, b = _aligner(loss_a, loss_b)
    d = (a - b).to_numpy()
    n = d.size
    if n < 3:
        return {"dm_stat": np.nan, "pval": np.nan, "mean_d": np.nan, "n": n, "lags": 0}
    lags = _nw_lags(n)
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1)) * np.dot(dc[k:], dc[:-k]) / n
    if lrv <= 0:
        return {"dm_stat": np.nan, "pval": np.nan, "mean_d": float(d.mean()), "n": n, "lags": lags}
    h = 1
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm = float(d.mean() / np.sqrt(lrv / n) * hln)
    return {"dm_stat": dm, "pval": float(2 * stats.t.sf(abs(dm), df=n - 1)),
            "mean_d": float(d.mean()), "n": n, "lags": lags}


def mincer_zarnowitz(realized_var: pd.Series, forecast_var: pd.Series) -> dict:
    """Régression de Mincer-Zarnowitz : RV_t = a + b · h_t + u_t.

    Une prévision sans biais vérifie a = 0 et b = 1. Écarts-types de
    Newey-West (HAC) ; test de Wald joint a = 0, b = 1 (loi F).
    Variances en %² par jour.

    Retourne un dict : a, b, se_a, se_b, r2, wald_stat, pval_joint, n.
    """
    rv, h = _aligner(realized_var, forecast_var)
    X = sm.add_constant(h.to_numpy())
    res = sm.OLS(rv.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": _nw_lags(len(rv))})
    wald = res.wald_test((np.eye(2), np.array([0.0, 1.0])), use_f=True, scalar=True)
    return {
        "a": float(res.params[0]), "b": float(res.params[1]),
        "se_a": float(res.bse[0]), "se_b": float(res.bse[1]),
        "r2": float(res.rsquared),
        "wald_stat": float(wald.statistic), "pval_joint": float(wald.pvalue),
        "n": int(len(rv)),
    }


def evaluer_volatilite(realized_var: pd.Series, forecast_var: pd.Series) -> dict:
    """Résumé de la qualité d'une variance prévue face à une variance réalisée.

    Retourne un dict : qlike_moyen, mse_moyen (%⁴), mz_a, mz_b, mz_r2, mz_p (test a=0,b=1), n.
    """
    pertes = vol_loss(realized_var, forecast_var)
    mz = mincer_zarnowitz(realized_var, forecast_var)
    return {"qlike_moyen": float(pertes["qlike"].mean()), "mse_moyen": float(pertes["mse"].mean()),
            "mz_a": mz["a"], "mz_b": mz["b"], "mz_r2": mz["r2"], "mz_p": mz["pval_joint"],
            "n": int(len(pertes))}
