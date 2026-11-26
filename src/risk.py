"""Module de risque D04 : volatilité conditionnelle, VaR et Expected Shortfall.

Ce module est consommé par le notebook D04, par l'application (vue Risk
Management, D07), par la stratégie et le backtest (D06) et par le paper trading
(D08). En conséquence :

- les fonctions renvoient des DataFrames ou des dict, jamais de print ni de
  graphique ;
- aucune fonction ne dépend d'une variable globale ni du notebook ;
- le modèle retenu (coûteux à estimer) est sauvegardé en JSON puis rechargé
  sans ré-estimation (``save_risk_model`` / ``load_risk_model``).

Unités (critère d'acceptation)
------------------------------
- Rendements : r_pct = 100 × rendement log journalier, en % par jour.
- Volatilité σ : en % par jour. Annualisée : σ × √252, toujours nommée
  explicitement (suffixe ``_annuelle_pct``).
- VaR et ES : PERTE POSITIVE en % par jour, sur une séance. En USD : perte sur
  le capital (100 000 $ par défaut, lu dans frozen.yaml).

Protocole anti-fuite
--------------------
1. Paramètres estimés sur l'apprentissage seul (``fit_vol_model``).
2. Paramètres gelés et appliqués à la série complète via
   ``arch_model(r_full, ...).fix(params)`` — aucune ré-estimation.
3. Prévision à un pas : ``fixed.forecast(horizon=1, start=r_full.index[0],
   reindex=False)``. La ligne d'origine t contient la prévision pour t+1 ; elle
   est donc rangée sous ``date_cible`` = séance suivante, avec
   ``date_prevision`` = t.
4. Les baselines (EWMA, glissante, simulation historique) n'utilisent, pour la
   date cible t+1, que des rendements datés ≤ t.
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from arch import arch_model
from arch import __version__ as _ARCH_VERSION
from arch.univariate import Normal, SkewStudent, StudentsT
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

from src.features import log_returns

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "configs" / "frozen.yaml"
DEFAULT_MODEL_PATH = _ROOT / "models" / "risk_params.json"

JOURS_PAR_AN = 252


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _risk_config(config: dict | None = None) -> dict:
    """Renvoie la section ``risk`` de la configuration.

    Accepte la configuration complète, la section ``risk`` seule, ou None
    (lecture de configs/frozen.yaml).
    """
    if config is None:
        config = _load_config()
    return config.get("risk", config)


def _niveau(alpha: float) -> int:
    """Niveau de confiance entier associé à alpha : 0.05 → 95, 0.01 → 99."""
    return int(round(100 * (1 - alpha)))


# ---------------------------------------------------------------------------
# Préparation des rendements
# ---------------------------------------------------------------------------

def rendements_pct(closes: pd.DataFrame | pd.Series, ticker: str | None = None) -> pd.Series:
    """Rendements log journaliers en % par jour : r_pct = 100 × ln(C_t / C_{t−1}).

    Contrairement à ``build_frame`` (qui supprime la dernière séance faute de
    cible), cette fonction conserve la DERNIÈRE séance du snapshot : c'est
    nécessaire pour prévoir le risque de la séance suivante (``risk_today``).

    Paramètres
    ----------
    closes : clôtures ajustées (DataFrame format large ou Series), index = date.
    ticker : colonne à utiliser si ``closes`` est un DataFrame (défaut : actif
             de frozen.yaml, soit DIA).

    Retourne une Series en % par jour, sans NaN, nommée ``r_pct``.
    """
    cfg = _load_config()
    scale = cfg["risk"]["scale"]
    if isinstance(closes, pd.DataFrame):
        closes = closes[ticker or cfg["tickers"]["asset"]]
    return (scale * log_returns(closes)).dropna().rename("r_pct")


def fenetres(serie: pd.Series | pd.DataFrame, config: dict | None = None) -> dict:
    """Découpe une série selon les fenêtres gelées (train, validation, test).

    Les bornes viennent de frozen.yaml ; aucune n'est déduite des données.
    Retourne un dict {nom_fenetre: sous-série}.
    """
    cfg = config if config is not None else _load_config()
    out = {}
    for nom, (debut, fin) in cfg["windows"].items():
        masque = (serie.index >= pd.Timestamp(debut)) & (serie.index <= pd.Timestamp(fin))
        out[nom] = serie.loc[masque].copy()
    return out


# ---------------------------------------------------------------------------
# Test ARCH-LM
# ---------------------------------------------------------------------------

def arch_lm_test(x: pd.Series, nlags: int = 5, seuil: float = 0.05) -> dict:
    """Test ARCH-LM d'Engle (1982) sur une série (rendements centrés ou résidus).

    H0 : pas d'effet ARCH (variance conditionnelle constante).
    Régression auxiliaire de x_t² sur ses ``nlags`` retards ; LM = n·R² ~ χ²(nlags).

    Unités : x en % par jour (la statistique est sans unité et invariante
    à l'échelle de x).

    Retourne un dict : nlags, stat_lm, pval_lm, stat_f, pval_f, n, conclusion.
    """
    x = pd.Series(x).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # changement d'API annoncé par statsmodels
        lm, lm_p, f, f_p = het_arch(x.values, nlags=nlags)
    effet = lm_p < seuil
    return {
        "nlags": nlags,
        "stat_lm": float(lm),
        "pval_lm": float(lm_p),
        "stat_f": float(f),
        "pval_f": float(f_p),
        "n": int(len(x)),
        "effet_arch": bool(effet),
        "conclusion": (
            f"H0 rejetée au seuil {seuil:.0%} : effet ARCH présent"
            if effet else
            f"H0 non rejetée au seuil {seuil:.0%} : pas d'effet ARCH détecté"
        ),
    }


# ---------------------------------------------------------------------------
# Lois des innovations
# ---------------------------------------------------------------------------

_DISTRIBUTIONS = {"normal": Normal, "t": StudentsT, "skewt": SkewStudent}
_DIST_PARAMS = {"normal": [], "t": ["nu"], "skewt": ["eta", "lambda"]}


def distribution(dist: str):
    """Instancie la loi STANDARDISÉE (moyenne 0, variance 1) d'arch : normal, t, skewt."""
    try:
        return _DISTRIBUTIONS[dist]()
    except KeyError as exc:
        raise ValueError(f"Loi inconnue : {dist!r} (attendu : {list(_DISTRIBUTIONS)})") from exc


def dist_params(dist: str, params: pd.Series | dict) -> list[float] | None:
    """Extrait les paramètres de forme de la loi depuis les paramètres du modèle.

    normal → None ; t → [nu] ; skewt → [eta, lambda] (Hansen, 1994).
    """
    noms = _DIST_PARAMS[dist]
    if not noms:
        return None
    return [float(params[n]) for n in noms]


# ---------------------------------------------------------------------------
# Modèles de volatilité
# ---------------------------------------------------------------------------

def _spec(vol: str, p: int, o: int, q: int, dist: str, name: str | None = None) -> dict:
    """Spécification sérialisable d'un modèle de volatilité."""
    if name is None:
        name = "GJR-GARCH" if (vol.upper() == "GARCH" and o > 0) else vol.upper()
    return {"name": name, "vol": vol, "p": int(p), "o": int(o), "q": int(q), "dist": dist}


def nom_modele(spec: dict) -> str:
    """Nom lisible d'un modèle : ``<name>-<dist>``, par exemple ``GJR-GARCH-skewt``."""
    return f"{spec['name']}-{spec['dist']}"


def _arch(r_pct: pd.Series, spec: dict):
    """Construit le modèle arch (moyenne constante, données déjà en %, pas de rescale)."""
    return arch_model(
        r_pct,
        mean="Constant",
        vol=spec["vol"],
        p=spec["p"],
        o=spec["o"],
        q=spec["q"],
        dist=spec["dist"],
        rescale=False,
    )


def _fix(r_pct: pd.Series, spec: dict, params: pd.Series | dict):
    """Applique des paramètres GELÉS au modèle arch construit sur ``r_pct``.

    ``.fix()`` lit les paramètres par POSITION : on les réordonne d'abord selon
    l'ordre propre à arch (un dict venu de l'app dans un autre ordre donnerait
    sinon des prévisions fausses sans erreur). Lève ValueError si les noms ne
    correspondent pas à la spécification.
    """
    params = pd.Series(params, dtype=float)
    modele = _arch(r_pct, spec)
    noms = (list(modele.parameter_names())
            + list(modele.volatility.parameter_names())
            + list(modele.distribution.parameter_names()))
    if len(noms) != len(params) or set(noms) != set(params.index):
        raise ValueError(
            f"Paramètres incompatibles avec {nom_modele(spec)} : "
            f"reçus {list(params.index)}, attendus {noms}"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return modele.fix(params.reindex(noms))


def persistance(spec: dict, params: pd.Series | dict) -> float:
    """Persistance de la variance conditionnelle (sans unité).

    - GARCH  : Σα + Σβ
    - GJR    : Σα + Σβ + Σγ/2 — le facteur 1/2 suppose une loi d'innovation
               SYMÉTRIQUE (P(z<0) = 1/2) ; pour skewt c'est une approximation.
    - EGARCH : Σβ (la récurrence porte sur ln σ²).
    """
    params = pd.Series(params)
    alpha = params.filter(regex=r"^alpha\[").sum()
    beta = params.filter(regex=r"^beta\[").sum()
    gamma = params.filter(regex=r"^gamma\[").sum()
    if spec["vol"].upper() == "EGARCH":
        return float(beta)
    return float(alpha + beta + gamma / 2.0)


def demi_vie(pers: float) -> float:
    """Demi-vie d'un choc de variance, en SÉANCES : ln(0,5) / ln(persistance).

    NaN si la persistance n'est pas dans ]0, 1[ (pas de retour à la moyenne).
    """
    if not (0.0 < pers < 1.0):
        return float("nan")
    return float(np.log(0.5) / np.log(pers))


def esperance_abs_z(dist: str, params: pd.Series | dict, n_grille: int = 20000) -> float:
    """E|z| sous la loi standardisée ``dist`` (sans unité), par grille de quantiles.

    E|z| = ∫₀¹ |F⁻¹(u)| du, approchée par la moyenne sur les points milieux
    d'une grille fine. Vaut √(2/π) ≈ 0,798 pour la loi normale, moins pour t/skewt.
    """
    u = (np.arange(n_grille) + 0.5) / n_grille
    return float(np.mean(np.abs(distribution(dist).ppf(u, dist_params(dist, params)))))


def vol_long_terme(spec: dict, params: pd.Series | dict, pers: float | None = None) -> float:
    """Volatilité de long terme, en % PAR JOUR.

    - GARCH / GJR : σ_LT = √(ω / (1 − persistance)) (variance inconditionnelle).
    - EGARCH      : niveau GÉOMÉTRIQUE σ_LT = exp(E[ln σ²] / 2), avec
                    E[ln σ²] = (ω + Σα (E|z| − √(2/π))) / (1 − Σβ).
                    arch centre |z| par √(2/π) (valeur normale) quelle que soit
                    la loi : la correction E|z| − √(2/π) est donc nécessaire
                    pour t et skewt. On ne renvoie pas √E[σ²] : sous t ou skewt,
                    E[exp(α|z|)] est infinie et la variance inconditionnelle de
                    l'EGARCH n'existe pas. Ce niveau est inférieur à √E[σ²]
                    (Jensen) — à comparer à ``vol_moyenne_apprentissage_pct``.
    NaN si la persistance est ≥ 1.
    """
    params = pd.Series(params)
    if pers is None:
        pers = persistance(spec, params)
    if not (pers < 1.0):
        return float("nan")
    omega = float(params["omega"])
    if spec["vol"].upper() == "EGARCH":
        alpha = float(params.filter(regex=r"^alpha\[").sum())
        correction = alpha * (esperance_abs_z(spec["dist"], params) - np.sqrt(2.0 / np.pi))
        return float(np.exp((omega + correction) / (2.0 * (1.0 - pers))))
    if omega <= 0:
        return float("nan")
    return float(np.sqrt(omega / (1.0 - pers)))


def fit_vol_model(
    r_pct_train: pd.Series,
    vol: str = "GARCH",
    p: int = 1,
    o: int = 0,
    q: int = 1,
    dist: str = "normal",
    name: str | None = None,
) -> dict:
    """Estime un modèle de volatilité à moyenne constante sur l'APPRENTISSAGE seul.

    r_t = μ + ε_t,  ε_t = σ_t z_t,  z_t ~ loi standardisée ``dist``.

    Paramètres
    ----------
    r_pct_train : rendements log en % par jour, fenêtre d'apprentissage uniquement.
    vol, p, o, q : spécification arch (vol="GARCH" et o=1 → GJR-GARCH ;
                   vol="EGARCH" → EGARCH de Nelson, 1991).
    dist        : "normal", "t" (Student standardisée) ou "skewt" (Hansen, 1994).
    name        : nom de la famille (défaut : déduit de vol et o).

    Retourne un dict :
    - spec : spécification sérialisable (name, vol, p, o, q, dist) ;
    - nom : ``<name>-<dist>`` ;
    - params, pvalues : pd.Series (μ en % par jour, ω en %², α, β, γ sans unité) ;
    - aic, bic, loglik : critères d'information et log-vraisemblance ;
    - persistance (sans unité), demi_vie_seances (séances),
      vol_lt_pct_jour (% par jour), vol_lt_annuelle_pct (% par an = × √252) ;
    - vol_moyenne_apprentissage_pct : √(moyenne de σ_t²) sur l'apprentissage,
      en % par jour (contrôle empirique de vol_lt_pct_jour) ;
    - temps_estimation_s : secondes (time.perf_counter) ;
    - convergence : code arch (0 = convergé), converge : bool ;
    - nobs, debut, fin : taille et bornes de l'échantillon ;
    - resultat : objet ARCHModelResult (résidus standardisés, diagnostics).
    """
    spec = _spec(vol, p, o, q, dist, name)
    r = pd.Series(r_pct_train).dropna()

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _arch(r, spec).fit(disp="off")
    t1 = time.perf_counter()

    pers = persistance(spec, res.params)
    vlt = vol_long_terme(spec, res.params, pers)
    return {
        "spec": spec,
        "nom": nom_modele(spec),
        "params": res.params.copy(),
        "pvalues": res.pvalues.copy(),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "loglik": float(res.loglikelihood),
        "persistance": pers,
        "demi_vie_seances": demi_vie(pers),
        "vol_lt_pct_jour": vlt,
        "vol_lt_annuelle_pct": vlt * np.sqrt(JOURS_PAR_AN),
        "vol_moyenne_apprentissage_pct": float(np.sqrt(np.mean(res.conditional_volatility ** 2))),
        "temps_estimation_s": t1 - t0,
        "convergence": int(res.convergence_flag),
        "converge": res.convergence_flag == 0,
        "nobs": int(res.nobs),
        "debut": r.index[0],
        "fin": r.index[-1],
        "resultat": res,
    }


def fit_grid(r_pct_train: pd.Series, config: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Estime toute la grille modèles × lois de frozen.yaml (3 × 3 = 9 combinaisons).

    Chaque combinaison est estimée sur l'apprentissage seul. Une estimation qui
    échoue ou ne converge pas reste dans le tableau (colonne ``converge``,
    ``erreur``) : rien n'est retiré en silence.

    Règle de choix (fixée avant de regarder la validation et le test) : le
    modèle retenu est celui de plus petit AIC parmi les modèles convergés.

    Retourne
    --------
    tableau : DataFrame trié par AIC croissant ; colonnes modele, famille, loi,
              aic, bic, loglik, persistance, demi_vie_seances (séances),
              vol_lt_pct_jour (% par jour), vol_lt_annuelle_pct (% par an),
              vol_moyenne_apprentissage_pct (% par jour),
              temps_estimation_s (s), converge, erreur.
    ajustements : dict {modele: résultat de ``fit_vol_model``}.
    """
    rcfg = _risk_config(config)
    lignes, ajustements = [], {}
    for m in rcfg["models"]:
        for dist in rcfg["distributions"]:
            spec = _spec(m["vol"], m["p"], m["o"], m["q"], dist, m["name"])
            nom = nom_modele(spec)
            try:
                fit = fit_vol_model(r_pct_train, spec["vol"], spec["p"], spec["o"],
                                    spec["q"], dist, name=spec["name"])
                ajustements[nom] = fit
                lignes.append({
                    "modele": nom, "famille": spec["name"], "loi": dist,
                    "aic": fit["aic"], "bic": fit["bic"], "loglik": fit["loglik"],
                    "persistance": fit["persistance"],
                    "demi_vie_seances": fit["demi_vie_seances"],
                    "vol_lt_pct_jour": fit["vol_lt_pct_jour"],
                    "vol_lt_annuelle_pct": fit["vol_lt_annuelle_pct"],
                    "vol_moyenne_apprentissage_pct": fit["vol_moyenne_apprentissage_pct"],
                    "temps_estimation_s": fit["temps_estimation_s"],
                    "converge": fit["converge"], "erreur": "",
                })
            except Exception as exc:  # l'échec est consigné, pas masqué
                lignes.append({
                    "modele": nom, "famille": spec["name"], "loi": dist,
                    "aic": np.nan, "bic": np.nan, "loglik": np.nan,
                    "persistance": np.nan, "demi_vie_seances": np.nan,
                    "vol_lt_pct_jour": np.nan, "vol_lt_annuelle_pct": np.nan,
                    "vol_moyenne_apprentissage_pct": np.nan,
                    "temps_estimation_s": np.nan, "converge": False,
                    "erreur": f"{type(exc).__name__}: {exc}",
                })
    tableau = (
        pd.DataFrame(lignes)
        .sort_values(["converge", "aic"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    return tableau, ajustements


def modele_retenu(tableau: pd.DataFrame) -> str:
    """Applique la règle fixée : plus petit AIC d'apprentissage parmi les modèles convergés."""
    ok = tableau[tableau["converge"] & tableau["aic"].notna()]
    if ok.empty:
        raise RuntimeError("Aucun modèle de la grille n'a convergé.")
    return str(ok.sort_values("aic").iloc[0]["modele"])


# ---------------------------------------------------------------------------
# Prévision de volatilité (paramètres gelés)
# ---------------------------------------------------------------------------

def seance_suivante(date: pd.Timestamp) -> pd.Timestamp:
    """Prochaine séance NYSE strictement après ``date`` (calendrier pandas_market_calendars).

    Repli sur le jour ouvré suivant si le calendrier est indisponible.
    """
    date = pd.Timestamp(date).normalize()
    try:
        import pandas_market_calendars as mcal
        jours = mcal.get_calendar("NYSE").valid_days(date + pd.Timedelta(days=1),
                                                     date + pd.Timedelta(days=15))
        return pd.Timestamp(jours[0]).tz_localize(None).normalize()
    except Exception:
        return date + pd.offsets.BDay(1)


def _dates_cibles(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Date cible de chaque origine : séance suivante de l'index ; pour la
    dernière origine, prochaine séance NYSE (inconnue des données)."""
    suivantes = list(index[1:]) + [seance_suivante(index[-1])]
    return pd.DatetimeIndex(suivantes, name="date_cible")


def forecast_vol(model_spec: dict, params: pd.Series | dict, r_pct_full: pd.Series) -> pd.DataFrame:
    """Prévisions à UN pas avec paramètres GELÉS, sur toute la série.

    1. ``arch_model(r_pct_full, ...).fix(params)`` — aucune ré-estimation.
    2. ``forecast(horizon=1, start=r_pct_full.index[0], reindex=False)`` : la
       ligne d'origine t ne dépend que de r_s pour s ≤ t et prévoit t+1.
    3. Décalage : la prévision est rangée sous ``date_cible`` = séance suivante.

    La dernière ligne prévoit la séance qui suit le dernier rendement connu ;
    sa date cible est la prochaine séance NYSE (pas encore observée).

    Les toutes premières prévisions dépendent de l'initialisation arch
    (backcast calculé sur le début de la série, donc sur l'apprentissage) ;
    l'effet disparaît en quelques dizaines de séances.

    Paramètres
    ----------
    model_spec : dict (vol, p, o, q, dist) — voir ``fit_vol_model``.
    params     : paramètres estimés sur l'apprentissage (pd.Series ou dict).
    r_pct_full : rendements en % par jour sur toute la période.

    Retourne un DataFrame indexé par ``date_cible`` :
    - date_prevision : date t de l'information utilisée ;
    - mu_pct    : moyenne conditionnelle prévue, en % par jour ;
    - sigma_pct : volatilité conditionnelle prévue, en % par jour.
    ``df.attrs["temps_prevision_s"]`` contient la durée du calcul (secondes).
    """
    r = pd.Series(r_pct_full).dropna()
    params = pd.Series(params, dtype=float)
    t0 = time.perf_counter()
    fixe = _fix(r, model_spec, params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prev = fixe.forecast(horizon=1, start=r.index[0], reindex=False)
    t1 = time.perf_counter()

    origines = pd.DatetimeIndex(prev.variance.index)
    out = pd.DataFrame({
        "date_prevision": origines,
        "mu_pct": prev.mean["h.1"].to_numpy(),
        "sigma_pct": np.sqrt(prev.variance["h.1"].to_numpy()),
    }, index=_dates_cibles(origines))
    out.attrs["temps_prevision_s"] = t1 - t0
    return out


def residus_standardises(model_spec: dict, params: pd.Series | dict, r_pct: pd.Series) -> pd.Series:
    """Résidus standardisés z_t = (r_t − μ) / σ_t avec paramètres gelés (sans unité).

    σ_t est la volatilité conditionnelle construite avec l'information ≤ t−1.
    """
    r = pd.Series(r_pct).dropna()
    fixe = _fix(r, model_spec, params)
    return (fixe.resid / fixe.conditional_volatility).rename("z")


# ---------------------------------------------------------------------------
# VaR et ES paramétriques
# ---------------------------------------------------------------------------

def var_es_parametric(mu, sigma, alpha: float, dist_obj, dist_params=None,
                      n_grille: int = 20000) -> dict:
    """VaR et ES paramétriques à une séance, en PERTE POSITIVE (% par jour).

    q      = F⁻¹(alpha) de la loi STANDARDISÉE (``dist_obj.ppf(alpha, dist_params)``) ;
    es_std = E[z | z ≤ q] ≈ moyenne de F⁻¹(u) sur une grille fine de u dans
             ]0, alpha[ (points milieux : u n'atteint jamais 0) ;
    VaR    = −(μ + σ × q) ;   ES = −(μ + σ × es_std).

    Paramètres
    ----------
    mu, sigma   : moyenne et volatilité conditionnelles en % par jour
                  (scalaires, ndarray ou Series alignées).
    alpha       : probabilité de queue (0.05 → VaR 95 %, 0.01 → VaR 99 %).
    dist_obj    : loi standardisée arch (Normal, StudentsT, SkewStudent).
    dist_params : paramètres de forme (None, [nu], [eta, lambda]).

    Retourne un dict : var_pct, es_pct (même type que mu/sigma), q_std, es_std.
    """
    q = float(np.asarray(dist_obj.ppf(alpha, dist_params)).ravel()[0])
    u = alpha * (np.arange(n_grille) + 0.5) / n_grille
    es_std = float(np.mean(dist_obj.ppf(u, dist_params)))
    var = -(mu + sigma * q)
    es = -(mu + sigma * es_std)
    if isinstance(var, pd.Series):
        var = var.rename(f"var{_niveau(alpha)}_pct")
        es = es.rename(f"es{_niveau(alpha)}_pct")
    return {"var_pct": var, "es_pct": es, "q_std": q, "es_std": es_std}


def perte_usd(perte_pct, capital: float):
    """Convertit une perte en rendement LOG (% par jour) en perte en USD sur ``capital``.

    perte_usd = capital × (1 − exp(−perte_pct / 100)) : conversion exacte du
    rendement log vers la variation de valeur du portefeuille.
    """
    return capital * (1.0 - np.exp(-perte_pct / 100.0))


# ---------------------------------------------------------------------------
# Baselines (données ≤ t uniquement, décalées d'un pas)
# ---------------------------------------------------------------------------

def ewma_sigma(r_pct: pd.Series, lam: float | None = None) -> pd.Series:
    """Volatilité EWMA RiskMetrics (1996), en % par jour, indexée par date cible.

    σ²_{t+1} = λ σ²_t + (1 − λ) r_t²  (moyenne supposée nulle),
    initialisée par r_0². La valeur à la date t n'utilise que r_s, s ≤ t−1.
    λ par défaut : ``risk.ewma_lambda`` de frozen.yaml.
    """
    if lam is None:
        lam = _risk_config()["ewma_lambda"]
    r = pd.Series(r_pct).dropna()
    var = (r ** 2).ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(var).shift(1).rename("sigma_ewma_pct")


def rolling_sigma(r_pct: pd.Series, window: int | None = None) -> pd.Series:
    """Écart-type glissant sur ``window`` séances, en % par jour, indexé par date cible.

    La valeur à la date t utilise r_{t−window}, …, r_{t−1}.
    ``window`` par défaut : ``risk.rolling_window`` de frozen.yaml.
    """
    if window is None:
        window = _risk_config()["rolling_window"]
    r = pd.Series(r_pct).dropna()
    return r.rolling(window).std(ddof=1).shift(1).rename(f"sigma_glissante{window}_pct")


def historical_sigma(r_pct: pd.Series) -> pd.Series:
    """Écart-type historique expansif, en % par jour, indexé par date cible.

    La valeur à la date t utilise tous les rendements r_s, s ≤ t−1.
    """
    r = pd.Series(r_pct).dropna()
    return r.expanding(min_periods=2).std(ddof=1).shift(1).rename("sigma_historique_pct")


def historical_simulation_var_es(r_pct: pd.Series, window: int | None = None,
                                 alpha: float = 0.05) -> pd.DataFrame:
    """VaR et ES par simulation historique sur ``window`` séances, en PERTE POSITIVE (% par jour).

    Pour la date cible t : échantillon r_{t−window}, …, r_{t−1} ;
    VaR = −quantile empirique d'ordre alpha ; ES = −moyenne des rendements ≤ ce quantile.
    ``window`` par défaut : ``risk.hs_window`` de frozen.yaml.

    Retourne un DataFrame indexé par date cible : var<niv>_pct, es<niv>_pct
    (NaN tant que ``window`` séances ne sont pas disponibles).
    """
    if window is None:
        window = _risk_config()["hs_window"]
    r = pd.Series(r_pct).dropna()
    niv = _niveau(alpha)
    var = np.full(len(r), np.nan)
    es = np.full(len(r), np.nan)
    if len(r) > window:
        fen = np.lib.stride_tricks.sliding_window_view(r.to_numpy(), window)[:-1]
        q = np.quantile(fen, alpha, axis=1)
        queue = np.where(fen <= q[:, None], fen, np.nan)
        var[window:] = -q
        es[window:] = -np.nanmean(queue, axis=1)
    return pd.DataFrame({f"var{niv}_pct": var, f"es{niv}_pct": es}, index=r.index)


def parkinson_sigma(high: pd.Series, low: pd.Series) -> pd.Series:
    """Volatilité RÉALISÉE de Parkinson (1980), en % par jour, datée du jour même.

    σ²_t = (ln(H_t / L_t))² / (4 ln 2) ;  σ_t en % = 100 × √σ²_t.
    Mesure du risque réalisé (pas une prévision), plus précise que |r_t| car
    elle exploite l'amplitude intrajournalière. À comparer à la prévision dont
    la date cible est t.
    """
    scale = _risk_config()["scale"]
    var = np.log(high / low) ** 2 / (4.0 * np.log(2.0))
    return (scale * np.sqrt(var)).rename("sigma_parkinson_pct")


# ---------------------------------------------------------------------------
# Sauvegarde / rechargement du modèle retenu
# ---------------------------------------------------------------------------

def save_risk_model(spec: dict, params: pd.Series | dict, path: str | Path | None = None,
                    periode_apprentissage: tuple | list | None = None) -> Path:
    """Sauvegarde le modèle retenu en JSON (par défaut models/risk_params.json).

    Contenu : type (name, vol, p, o, q), loi, paramètres gelés (μ en % par jour,
    ω en %², autres sans unité), échelle des rendements (100 = %), date
    d'estimation, période d'apprentissage (frozen.yaml si non fournie) et
    version d'arch. Les flottants sont écrits en représentation exacte : le
    rechargement redonne les mêmes paramètres au bit près.

    Retourne le chemin écrit.
    """
    path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    cfg = _load_config()
    if periode_apprentissage is None:
        periode_apprentissage = cfg["windows"]["train"]
    contenu = {
        "spec": {k: spec[k] for k in ("name", "vol", "p", "o", "q", "dist")},
        "nom": nom_modele(spec),
        "params": {k: float(v) for k, v in pd.Series(params, dtype=float).items()},
        "scale": cfg["risk"]["scale"],
        "unites": {
            "rendements": "% par jour (100 x rendement log)",
            "mu": "% par jour",
            "omega": "%^2 par jour (EGARCH : ln %^2)",
        },
        "date_estimation": datetime.now().isoformat(timespec="seconds"),
        "periode_apprentissage": [str(pd.Timestamp(d).date()) for d in periode_apprentissage],
        "arch_version": _ARCH_VERSION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(contenu, fh, ensure_ascii=False, indent=2)
    return path


def load_risk_model(path: str | Path | None = None) -> dict:
    """Recharge un modèle sauvegardé par ``save_risk_model`` (sans ré-estimation).

    Retourne un dict : spec, nom, params (pd.Series), scale,
    date_estimation, periode_apprentissage, arch_version.
    """
    path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    with open(path, encoding="utf-8") as fh:
        contenu = json.load(fh)
    contenu["params"] = pd.Series(contenu["params"], dtype=float)
    return contenu


# ---------------------------------------------------------------------------
# Interface pour l'application, la stratégie et le paper trading
# ---------------------------------------------------------------------------

def risk_series(r_pct_full: pd.Series, model_path: str | Path | dict | None = None,
                capital: float | None = None) -> pd.DataFrame:
    """Risque prévu pour TOUTES les dates cibles, avec le modèle sauvegardé.

    Utilisée par le backtest (D06) et les graphiques de l'application (D07).

    Paramètres
    ----------
    r_pct_full : rendements en % par jour (``rendements_pct``).
    model_path : chemin du JSON (défaut models/risk_params.json) ou dict déjà
                 chargé par ``load_risk_model``.
    capital    : capital en USD (défaut : ``risk.capital`` de frozen.yaml).

    Retourne un DataFrame indexé par ``date_cible`` (séance prévue) :
    - date_prevision : dernière date d'information utilisée ;
    - mu_pct, sigma_pct : % par jour ; sigma_annuelle_pct : σ × √252, % par an ;
    - var95_pct, es95_pct, var99_pct, es99_pct : perte positive, % par jour ;
    - var95_usd, es95_usd, var99_usd, es99_usd : perte positive, USD sur ``capital``.
    La dernière ligne concerne la séance qui suit le dernier rendement connu.
    """
    rcfg = _risk_config()
    if capital is None:
        capital = rcfg["capital"]
    modele = model_path if isinstance(model_path, dict) else load_risk_model(model_path)
    spec, params = modele["spec"], pd.Series(modele["params"], dtype=float)

    prev = forecast_vol(spec, params, r_pct_full)
    out = prev[["date_prevision", "mu_pct", "sigma_pct"]].copy()
    out["sigma_annuelle_pct"] = out["sigma_pct"] * np.sqrt(JOURS_PAR_AN)

    loi, forme = distribution(spec["dist"]), dist_params(spec["dist"], params)
    for alpha in rcfg["alphas"]:
        niv = _niveau(alpha)
        r = var_es_parametric(out["mu_pct"], out["sigma_pct"], alpha, loi, forme)
        out[f"var{niv}_pct"] = r["var_pct"]
        out[f"es{niv}_pct"] = r["es_pct"]
    for alpha in rcfg["alphas"]:
        niv = _niveau(alpha)
        out[f"var{niv}_usd"] = perte_usd(out[f"var{niv}_pct"], capital)
        out[f"es{niv}_usd"] = perte_usd(out[f"es{niv}_pct"], capital)
    out.attrs["temps_prevision_s"] = prev.attrs.get("temps_prevision_s")
    out.attrs["modele"] = modele["nom"]
    out.attrs["capital"] = capital
    return out


def risk_today(r_pct_full: pd.Series, model_path: str | Path | dict | None = None,
               capital: float | None = None) -> dict:
    """Risque prévu pour la séance t+1, où t est la DERNIÈRE date de ``r_pct_full``.

    Utilisée par l'application (D07), la stratégie (D06) et le paper trading (D08).
    Identique, par construction, à la dernière ligne de ``risk_series``.

    Retourne un dict : modele, capital, date_prevision, date_cible, mu_pct,
    sigma_pct (% par jour), sigma_annuelle_pct (% par an), var95_pct, es95_pct,
    var99_pct, es99_pct (perte positive, % par jour), var95_usd, es95_usd,
    var99_usd, es99_usd (perte positive, USD sur ``capital``).
    """
    serie = risk_series(r_pct_full, model_path, capital)
    derniere = serie.iloc[-1]
    out = {
        "modele": serie.attrs["modele"],
        "capital": float(serie.attrs["capital"]),
        "date_prevision": pd.Timestamp(derniere["date_prevision"]),
        "date_cible": pd.Timestamp(serie.index[-1]),
    }
    for col in serie.columns:
        if col != "date_prevision":
            out[col] = float(derniere[col])
    return out


# ---------------------------------------------------------------------------
# Tableaux d'analyse (notebook D04, vue Risk Management de l'application)
# ---------------------------------------------------------------------------

def tableau_arch_lm(series: dict, lags=(5, 10)) -> pd.DataFrame:
    """Test ARCH-LM (Engle, 1982) pour plusieurs séries et plusieurs retards.

    Paramètres
    ----------
    series : dict {libellé: Series en % par jour} (rendements centrés, résidus…).
    lags   : retards testés.

    Retourne un DataFrame : serie, retards, stat_lm, p_value, n, conclusion.
    """
    lignes = []
    for nom, x in series.items():
        for k in lags:
            t = arch_lm_test(x, nlags=k)
            lignes.append({"serie": nom, "retards": k, "stat_lm": t["stat_lm"],
                           "p_value": t["pval_lm"], "n": t["n"], "conclusion": t["conclusion"]})
    return pd.DataFrame(lignes)


def correlation_levier(r_pct: pd.Series) -> dict:
    """Effet de levier : le rendement d'aujourd'hui annonce-t-il la volatilité de demain ?

    Corrélations de Pearson (avec p-value) entre r_t et |r_{t+1}|, et entre r_t
    et r²_{t+1}. Une corrélation NÉGATIVE signifie qu'une baisse est suivie
    d'une volatilité plus forte qu'une hausse de même taille (asymétrie que
    GJR-GARCH et EGARCH modélisent).
    Moyennes de |r_{t+1}| après une baisse et après une hausse, en % par jour.
    """
    r = pd.Series(r_pct).dropna()
    x, y = r.iloc[:-1].to_numpy(), r.iloc[1:].to_numpy()
    c1, p1 = scipy_stats.pearsonr(x, np.abs(y))
    c2, p2 = scipy_stats.pearsonr(x, y ** 2)
    return {
        "n": int(x.size),
        "corr_r_absr_suivant": float(c1), "pval_absr": float(p1),
        "corr_r_r2_suivant": float(c2), "pval_r2": float(p2),
        "absr_moyen_apres_baisse_pct": float(np.abs(y[x < 0]).mean()),
        "absr_moyen_apres_hausse_pct": float(np.abs(y[x > 0]).mean()),
    }


_ROLES = {
    "mu": ("moyenne conditionnelle", "% par jour"),
    "omega": ("constante de variance", "%² par jour (EGARCH : ln %²)"),
    "alpha[1]": ("réaction aux chocs", "sans unité"),
    "gamma[1]": ("asymétrie (effet de levier)", "sans unité"),
    "beta[1]": ("mémoire de la volatilité", "sans unité"),
    "nu": ("degrés de liberté (queues épaisses)", "sans unité"),
    "eta": ("degrés de liberté (queues épaisses)", "sans unité"),
    "lambda": ("asymétrie de la loi (Hansen)", "sans unité"),
}


def tableau_parametres(fit: dict) -> pd.DataFrame:
    """Paramètres estimés d'un modèle (``fit_vol_model``) avec p-values, rôle et unité."""
    lignes = []
    for nom, val in fit["params"].items():
        role, unite = _ROLES.get(nom, ("", ""))
        lignes.append({"parametre": nom, "valeur": float(val),
                       "p_value": float(fit["pvalues"][nom]), "role": role, "unite": unite})
    return pd.DataFrame(lignes)


def resume_modele(fit: dict) -> pd.DataFrame:
    """Indicateurs dérivés d'un modèle : persistance, demi-vie, volatilité de long terme."""
    rac = np.sqrt(JOURS_PAR_AN)
    lignes = [
        ("persistance", fit["persistance"], "sans unité"),
        ("demi-vie d'un choc", fit["demi_vie_seances"], "séances"),
        ("volatilité de long terme", fit["vol_lt_pct_jour"], "% par jour"),
        ("volatilité de long terme annualisée (× √252)", fit["vol_lt_annuelle_pct"], "% par an"),
        ("volatilité conditionnelle moyenne (apprentissage)",
         fit["vol_moyenne_apprentissage_pct"], "% par jour"),
        ("volatilité conditionnelle moyenne annualisée (× √252)",
         fit["vol_moyenne_apprentissage_pct"] * rac, "% par an"),
    ]
    return pd.DataFrame(lignes, columns=["indicateur", "valeur", "unite"])


def diagnostics_residus_std(z: pd.Series, lags=(5, 10, 20), arch_lags=(5, 10)) -> pd.DataFrame:
    """Diagnostics des résidus standardisés z (sans unité).

    Ljung-Box sur z (autocorrélation restante de la moyenne) et sur z²
    (effet ARCH restant), ARCH-LM sur z, Jarque-Bera (normalité ; un rejet est
    attendu si la loi retenue n'est pas normale). H0 = « résidus propres ».

    Retourne un DataFrame : test, retards, stat, p_value, conclusion (seuil
    ``risk.seuil_tests``). ``df.attrs`` : asymetrie et kurtosis de z.
    """
    seuil = _risk_config()["seuil_tests"]
    z = pd.Series(z).dropna()
    lignes = []
    lb = acorr_ljungbox(z, lags=list(lags), return_df=True)
    lb2 = acorr_ljungbox(z ** 2, lags=list(lags), return_df=True)
    for k in lags:
        lignes.append(("Ljung-Box z", k, lb.loc[k, "lb_stat"], lb.loc[k, "lb_pvalue"]))
    for k in lags:
        lignes.append(("Ljung-Box z²", k, lb2.loc[k, "lb_stat"], lb2.loc[k, "lb_pvalue"]))
    for k in arch_lags:
        t = arch_lm_test(z, nlags=k)
        lignes.append(("ARCH-LM z", k, t["stat_lm"], t["pval_lm"]))
    jb, jb_p, asym, kurt = jarque_bera(z)
    lignes.append(("Jarque-Bera z", np.nan, jb, jb_p))
    df = pd.DataFrame(lignes, columns=["test", "retards", "stat", "p_value"])
    df["conclusion"] = np.where(df["p_value"] < seuil, "H0 rejetée", "H0 non rejetée")
    df.attrs["asymetrie"] = float(asym)
    df.attrs["kurtosis"] = float(kurt)
    return df


def qq_points(z: pd.Series, dist: str, params: pd.Series | dict) -> pd.DataFrame:
    """Points du QQ-plot des résidus standardisés contre la loi STANDARDISÉE ``dist``.

    Quantiles théoriques F⁻¹((i − 0,5)/n) contre quantiles empiriques triés (sans unité).
    """
    zs = np.sort(pd.Series(z).dropna().to_numpy())
    p = (np.arange(1, zs.size + 1) - 0.5) / zs.size
    theo = np.asarray(distribution(dist).ppf(p, dist_params(dist, params)))
    return pd.DataFrame({"theorique": theo, "empirique": zs})


# ---------------------------------------------------------------------------
# Mesures de risque comparées (modèle retenu, GARCH normal, EWMA, HS)
# ---------------------------------------------------------------------------

def _ajouter_usd(serie: pd.DataFrame, capital: float, alphas) -> pd.DataFrame:
    for alpha in alphas:
        niv = _niveau(alpha)
        serie[f"var{niv}_usd"] = perte_usd(serie[f"var{niv}_pct"], capital)
        serie[f"es{niv}_usd"] = perte_usd(serie[f"es{niv}_pct"], capital)
    return serie


def mesure_parametrique(r_pct: pd.Series, modele: dict, nom: str | None = None,
                        capital: float | None = None) -> dict:
    """Mesure de risque d'un modèle GARCH/GJR/EGARCH à paramètres gelés.

    ``modele`` : dict avec spec, params, nom (``fit_vol_model`` ou ``load_risk_model``).
    Retourne un dict : nom, type="parametrique", serie (``risk_series``, % par
    jour et USD), dist, dist_params.
    """
    serie = risk_series(r_pct, modele, capital)
    spec = modele["spec"]
    return {"nom": nom or modele["nom"], "type": "parametrique", "serie": serie,
            "dist": spec["dist"], "dist_params": dist_params(spec["dist"], modele["params"])}


def mesure_ewma(r_pct: pd.Series, lam: float | None = None, capital: float | None = None) -> dict:
    """Mesure de risque EWMA RiskMetrics : loi normale, μ = 0, σ = ``ewma_sigma``.

    VaR/ES en perte positive (% par jour et USD), indexées par date cible ;
    seules les données ≤ t sont utilisées pour la date cible t+1.
    """
    rcfg = _risk_config()
    capital = rcfg["capital"] if capital is None else capital
    sigma = ewma_sigma(r_pct, lam)
    serie = pd.DataFrame({"mu_pct": 0.0, "sigma_pct": sigma}, index=sigma.index)
    serie["sigma_annuelle_pct"] = serie["sigma_pct"] * np.sqrt(JOURS_PAR_AN)
    loi = distribution("normal")
    for alpha in rcfg["alphas"]:
        v = var_es_parametric(serie["mu_pct"], serie["sigma_pct"], alpha, loi, None)
        serie[f"var{_niveau(alpha)}_pct"] = v["var_pct"]
        serie[f"es{_niveau(alpha)}_pct"] = v["es_pct"]
    serie = _ajouter_usd(serie, capital, rcfg["alphas"]).dropna()
    return {"nom": "EWMA", "type": "parametrique", "serie": serie,
            "dist": "normal", "dist_params": None}


def mesure_historique(r_pct: pd.Series, window: int | None = None,
                      capital: float | None = None) -> dict:
    """Mesure de risque par simulation historique sur ``window`` séances (défaut : hs_window).

    VaR/ES en perte positive (% par jour et USD), indexées par date cible.
    """
    rcfg = _risk_config()
    capital = rcfg["capital"] if capital is None else capital
    window = rcfg["hs_window"] if window is None else window
    serie = pd.concat([historical_simulation_var_es(r_pct, window, a) for a in rcfg["alphas"]],
                      axis=1)
    serie = _ajouter_usd(serie, capital, rcfg["alphas"]).dropna()
    return {"nom": f"Simulation historique ({window} j)", "type": "historique",
            "serie": serie, "window": window}


def tableau_risque_date(r_pct: pd.Series, serie: pd.DataFrame, sigma_parkinson: pd.Series,
                        fenetres_idx: dict) -> pd.DataFrame:
    """Tableau daté prévision / réalisation du modèle retenu (export risque_date.csv).

    Paramètres
    ----------
    r_pct           : rendements réalisés en % par jour.
    serie           : sortie de ``risk_series`` (indexée par date cible).
    sigma_parkinson : volatilité réalisée de Parkinson en % par jour.
    fenetres_idx    : dict {nom_fenetre: DatetimeIndex des dates cibles}.

    Colonnes : date_prevision, date_cible, fenetre, rendement_realise_pct,
    sigma_pct, sigma_parkinson_pct, var95_pct, es95_pct, var99_pct, es99_pct
    (% par jour), var95_usd (USD), violation95, violation99 (1 si r < −VaR).
    """
    morceaux = []
    for nom, idx in fenetres_idx.items():
        s = serie.reindex(idx)
        df = pd.DataFrame({
            "date_prevision": s["date_prevision"],
            "date_cible": idx,
            "fenetre": nom,
            "rendement_realise_pct": r_pct.reindex(idx),
            "sigma_pct": s["sigma_pct"],
            "sigma_parkinson_pct": sigma_parkinson.reindex(idx),
            "var95_pct": s["var95_pct"], "es95_pct": s["es95_pct"],
            "var99_pct": s["var99_pct"], "es99_pct": s["es99_pct"],
            "var95_usd": s["var95_usd"],
        }, index=idx)
        df["violation95"] = (df["rendement_realise_pct"] < -df["var95_pct"]).astype(int)
        df["violation99"] = (df["rendement_realise_pct"] < -df["var99_pct"]).astype(int)
        morceaux.append(df)
    return pd.concat(morceaux).reset_index(drop=True)


def resume_var_moyenne(serie: pd.DataFrame, fenetres_idx: dict) -> pd.DataFrame:
    """VaR et ES moyennes par fenêtre, en % par jour et en USD (perte positive, 1 séance).

    ``serie`` : sortie de ``risk_series`` ou ``mesure_*``["serie"].
    Retourne un DataFrame : fenetre, mesure, niveau, moyenne_pct, moyenne_usd, n.
    """
    lignes = []
    for nom, idx in fenetres_idx.items():
        s = serie.reindex(idx)
        for alpha in _risk_config()["alphas"]:
            niv = _niveau(alpha)
            for mesure in ("var", "es"):
                lignes.append({
                    "fenetre": nom, "mesure": mesure.upper(), "niveau": niv,
                    "moyenne_pct": float(s[f"{mesure}{niv}_pct"].mean()),
                    "moyenne_usd": float(s[f"{mesure}{niv}_usd"].mean()),
                    "n": int(s[f"{mesure}{niv}_pct"].notna().sum()),
                })
    return pd.DataFrame(lignes)


def tableau_temps(ajustements: dict, r_pct: pd.Series) -> pd.DataFrame:
    """Coût de calcul (axe B09), en secondes.

    Pour chaque modèle de la grille : temps d'estimation sur l'apprentissage
    (``fit_vol_model``) et temps de prévision 1 pas sur la série complète
    (``forecast_vol``). Les baselines (EWMA, glissante, simulation historique)
    n'ont pas d'estimation : seul le temps de calcul de la série est mesuré.
    """
    lignes = []
    for nom, fit in ajustements.items():
        prev = forecast_vol(fit["spec"], fit["params"], r_pct)
        lignes.append({"modele": nom, "estimation_s": fit["temps_estimation_s"],
                       "prevision_s": prev.attrs["temps_prevision_s"]})
    rcfg = _risk_config()
    baselines = {
        "EWMA": lambda: ewma_sigma(r_pct),
        f"Glissante {rcfg['rolling_window']} j": lambda: rolling_sigma(r_pct),
        f"Simulation historique ({rcfg['hs_window']} j)":
            lambda: [historical_simulation_var_es(r_pct, alpha=a) for a in rcfg["alphas"]],
    }
    for nom, calcul in baselines.items():
        t0 = time.perf_counter()
        calcul()
        lignes.append({"modele": nom, "estimation_s": 0.0,
                       "prevision_s": time.perf_counter() - t0})
    df = pd.DataFrame(lignes)
    df["total_s"] = df["estimation_s"] + df["prevision_s"]
    return df.sort_values("total_s", ascending=False).reset_index(drop=True)
