"""Tests du module de risque D04 : anti-fuite, unités, backtests, sauvegarde."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.risk import (
    distribution, ewma_sigma, fit_vol_model, forecast_vol,
    historical_simulation_var_es, load_risk_model, rendements_pct, risk_series,
    risk_today, rolling_sigma, save_risk_model, seance_suivante, var_es_parametric,
)
from src.risk_backtest import (
    christoffersen_test, dm_test_loss, es_backtest_acerbi_szekely, kupiec_test,
    mincer_zarnowitz, simulateur_parametrique, vol_loss,
)


# ---------------------------------------------------------------------------
# Données synthétiques : prix OHLC à volatilité GARCH
# ---------------------------------------------------------------------------

def _ohlc_synthetiques(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Prix OHLC synthétiques dont les rendements suivent un GARCH(1,1) normal."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.02, 0.08, 0.90          # variance en %² par jour
    s2 = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    dates = pd.bdate_range("2020-01-02", periods=n, name="date")
    close = 100.0 * np.exp(np.cumsum(r / 100.0))
    open_ = np.r_[100.0, close[:-1]]
    ecart = np.abs(rng.normal(0, 0.004, n))
    return pd.DataFrame({
        "open": open_, "close": close,
        "high": np.maximum(open_, close) * (1 + ecart),
        "low": np.minimum(open_, close) * (1 - ecart),
    }, index=dates)


def _falsifier(ohlc: pd.DataFrame, cut: pd.Timestamp, facteur: float) -> pd.DataFrame:
    """Multiplie les PRIX BRUTS par ``facteur`` à partir de la date ``cut`` incluse."""
    faux = ohlc.copy()
    faux.loc[faux.index >= cut, ["open", "high", "low", "close"]] *= facteur
    return faux


_SPEC_GARCH = {"name": "GARCH", "vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "normal"}


def _sigma_garch(ohlc: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Chaîne complète : prix → rendements % → estimation (250 premières séances) → σ prévue."""
    r = rendements_pct(ohlc["close"])
    fit = fit_vol_model(r.iloc[:250], "GARCH", 1, 0, 1, "normal")
    return forecast_vol(fit["spec"], fit["params"], r)["sigma_pct"], fit["params"]


# ---------------------------------------------------------------------------
# 1. Anti-fuite : falsification des prix à partir d'une date cut
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def donnees_cut():
    ohlc = _ohlc_synthetiques()
    cut = ohlc.index[400]
    return ohlc, cut, _falsifier(ohlc, cut, 1.5)


def _avant_apres(a: pd.Series, b: pd.Series, cut: pd.Timestamp, n_apres: int = 5):
    """Vérifie : identiques pour date cible ≤ cut, différents juste après."""
    a, b = a.dropna(), b.dropna()
    avant_a, avant_b = a[a.index <= cut], b[b.index <= cut]
    assert avant_a.index.equals(avant_b.index)
    assert np.array_equal(avant_a.to_numpy(), avant_b.to_numpy()), "Fuite : valeurs ≤ cut modifiées"
    apres_a, apres_b = a[a.index > cut].iloc[:n_apres], b[b.index > cut].iloc[:n_apres]
    assert len(apres_a) == n_apres
    assert (apres_a.to_numpy() != apres_b.to_numpy()).all(), "La falsification devrait se propager après cut"


def test_antifuite_garch(donnees_cut):
    """σ GARCH prévue (paramètres gelés) : identique jusqu'à cut inclus, différente ensuite."""
    ohlc, cut, faux = donnees_cut
    sig_vrai, p_vrai = _sigma_garch(ohlc)
    sig_faux, p_faux = _sigma_garch(faux)
    assert np.array_equal(p_vrai.to_numpy(), p_faux.to_numpy()), "Paramètres d'apprentissage modifiés"
    _avant_apres(sig_vrai, sig_faux, cut)


def test_antifuite_ewma_et_glissante(donnees_cut):
    """EWMA et écart-type glissant : aucune donnée ≥ date cible utilisée."""
    ohlc, cut, faux = donnees_cut
    r_vrai, r_faux = rendements_pct(ohlc["close"]), rendements_pct(faux["close"])
    _avant_apres(ewma_sigma(r_vrai), ewma_sigma(r_faux), cut)
    _avant_apres(rolling_sigma(r_vrai), rolling_sigma(r_faux), cut)


def test_antifuite_simulation_historique(donnees_cut):
    """Simulation historique : identique jusqu'à cut inclus.

    Avec ×1,5 le rendement falsifié est un GAIN : il n'entre pas dans la queue
    gauche et la VaR HS ne bouge pas ensuite. On vérifie donc la propagation
    après cut avec une falsification ×0,5 (krach), qui entre dans la queue.
    """
    ohlc, cut, faux = donnees_cut
    r_vrai = rendements_pct(ohlc["close"])
    hs_vrai = historical_simulation_var_es(r_vrai, alpha=0.05)
    hs_faux = historical_simulation_var_es(rendements_pct(faux["close"]), alpha=0.05)
    for col in ("var95_pct", "es95_pct"):
        a, b = hs_vrai[col].dropna(), hs_faux[col].dropna()
        assert np.array_equal(a[a.index <= cut].to_numpy(), b[b.index <= cut].to_numpy())

    krach = rendements_pct(_falsifier(ohlc, cut, 0.5)["close"])
    hs_krach = historical_simulation_var_es(krach, alpha=0.05)
    for col in ("var95_pct", "es95_pct"):
        _avant_apres(hs_vrai[col], hs_krach[col], cut)


# ---------------------------------------------------------------------------
# 2. Unités et cohérence VaR / ES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist,forme", [
    ("normal", None), ("t", [5.0]), ("skewt", [6.0, -0.2]),
])
def test_unites_var_es(dist, forme):
    """VaR > 0, ES ≥ VaR, VaR 99 % ≥ VaR 95 %, VaR croissante avec σ (perte positive en %)."""
    loi = distribution(dist)
    sigma = np.array([0.5, 1.0, 2.0, 4.0])       # % par jour
    mu = 0.02                                     # % par jour
    v95 = var_es_parametric(mu, sigma, 0.05, loi, forme)
    v99 = var_es_parametric(mu, sigma, 0.01, loi, forme)
    for v in (v95, v99):
        assert (v["var_pct"] > 0).all()
        assert (v["es_pct"] >= v["var_pct"]).all()
        assert (np.diff(v["var_pct"]) > 0).all()
    assert (v99["var_pct"] >= v95["var_pct"]).all()


def test_unites_risk_series(tmp_path):
    """risk_series : colonnes positives, σ annualisée = σ × √252, USD cohérents."""
    r = rendements_pct(_ohlc_synthetiques()["close"])
    fit = fit_vol_model(r.iloc[:250], "GARCH", 1, 0, 1, "t")
    chemin = save_risk_model(fit["spec"], fit["params"], tmp_path / "risk_params.json")
    s = risk_series(r, chemin, capital=100_000)
    assert np.allclose(s["sigma_annuelle_pct"], s["sigma_pct"] * np.sqrt(252))
    for niv in (95, 99):
        assert (s[f"var{niv}_pct"] > 0).all()
        assert (s[f"es{niv}_pct"] >= s[f"var{niv}_pct"]).all()
        assert (s[f"var{niv}_usd"] > 0).all()
        assert (s[f"var{niv}_usd"] < 100_000 * s[f"var{niv}_pct"] / 100).all()  # 1 − e^(−x) < x
    assert (s["var99_pct"] >= s["var95_pct"]).all()


# ---------------------------------------------------------------------------
# 3. Kupiec
# ---------------------------------------------------------------------------

def test_kupiec():
    """5 % de violations → H0 non rejetée ; 20 % → H0 rejetée (VaR 95 %)."""
    n = 1000
    bon = np.zeros(n, dtype=int); bon[::20] = 1            # 50 violations = 5 %
    mauvais = np.zeros(n, dtype=int); mauvais[::5] = 1     # 200 violations = 20 %
    assert kupiec_test(bon, 0.05)["pval"] > 0.05
    assert kupiec_test(mauvais, 0.05)["pval"] < 0.05


# ---------------------------------------------------------------------------
# 4. Christoffersen
# ---------------------------------------------------------------------------

def test_christoffersen_violations_groupees():
    """Un bloc de violations consécutives → indépendance rejetée."""
    hits = np.zeros(1000, dtype=int)
    hits[500:550] = 1                                       # 5 % mais groupées
    res = christoffersen_test(hits, alpha=0.05)
    assert kupiec_test(hits, 0.05)["pval"] > 0.05          # le taux seul est bon
    assert res["pval_ind"] < 0.05
    assert res["pval_cc"] < 0.05
    assert np.isclose(res["lr_cc"], res["lr_uc"] + res["lr_ind"])


# ---------------------------------------------------------------------------
# 5. Lois
# ---------------------------------------------------------------------------

def test_student_plus_prudente_que_normale_a_99():
    """À σ égal, la VaR 99 % Student (ν = 5) dépasse la VaR 99 % normale."""
    v_t = var_es_parametric(0.0, 1.0, 0.01, distribution("t"), [5.0])["var_pct"]
    v_n = var_es_parametric(0.0, 1.0, 0.01, distribution("normal"), None)["var_pct"]
    assert v_t > v_n
    assert np.isclose(v_n, 2.3263478740, atol=1e-8)


# ---------------------------------------------------------------------------
# 6. QLIKE, Diebold-Mariano, Mincer-Zarnowitz
# ---------------------------------------------------------------------------

def _variances_synthetiques(n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    s2 = pd.Series(np.exp(rng.normal(0.0, 0.5, n)), index=idx)     # %² par jour
    rv = s2 * rng.standard_normal(n) ** 2
    return s2, rv


def test_qlike_prevision_parfaite_meilleure():
    """La vraie variance a une QLIKE moyenne plus faible qu'une prévision fausse."""
    s2, rv = _variances_synthetiques()
    parfaite = vol_loss(rv, s2)
    for faux in (4.0 * s2, 0.25 * s2):
        fausse = vol_loss(rv, faux)
        assert parfaite["qlike"].mean() < fausse["qlike"].mean()
        dm = dm_test_loss(parfaite["qlike"], fausse["qlike"])
        assert dm["mean_d"] < 0 and dm["pval"] < 0.05


def test_mincer_zarnowitz_detecte_biais():
    """Prévision divisée par 2 → b ≈ 2, test joint a = 0, b = 1 rejeté."""
    s2, rv = _variances_synthetiques()
    mz = mincer_zarnowitz(rv, 0.5 * s2)
    assert mz["b"] > 1.5 and mz["pval_joint"] < 0.05


# ---------------------------------------------------------------------------
# Acerbi-Székely : un ES sous-estimé est détecté
# ---------------------------------------------------------------------------

def test_acerbi_szekely_es_optimiste():
    """σ prévue = moitié de la vraie → Z2 < 0 et p < 0,05."""
    rng = np.random.default_rng(42)
    n = 1000
    idx = pd.bdate_range("2020-01-02", periods=n)
    vrai_sigma = pd.Series(1.0, index=idx)
    r = pd.Series(rng.standard_normal(n), index=idx) * vrai_sigma
    loi = distribution("normal")
    sigma_prevue = 0.5 * vrai_sigma
    v = var_es_parametric(0.0, sigma_prevue, 0.05, loi, None)
    sim = simulateur_parametrique(np.zeros(n), sigma_prevue.to_numpy(), loi, None)
    res = es_backtest_acerbi_szekely(r, v["var_pct"], v["es_pct"], 0.05, sim, n_sim=2000)
    assert res["z2"] < 0 and res["pval"] < 0.05
    assert res["perte_moyenne_realisee_pct"] > res["es_moyen_prevu_pct"]


# ---------------------------------------------------------------------------
# 7. Sauvegarde et interface application
# ---------------------------------------------------------------------------

def test_sauvegarde_et_risk_today(tmp_path):
    """save → load redonne exactement les paramètres ; risk_today = dernière ligne de risk_series."""
    r = rendements_pct(_ohlc_synthetiques()["close"])
    fit = fit_vol_model(r.iloc[:250], "GARCH", 1, 1, 1, "skewt", name="GJR-GARCH")
    chemin = save_risk_model(fit["spec"], fit["params"], tmp_path / "risk_params.json")
    modele = load_risk_model(chemin)

    assert list(modele["params"].index) == list(fit["params"].index)
    assert np.array_equal(modele["params"].to_numpy(), fit["params"].to_numpy())
    assert modele["spec"] == fit["spec"]

    serie = risk_series(r, chemin)
    today = risk_today(r, chemin)
    derniere = serie.iloc[-1]
    assert today["date_prevision"] == r.index[-1]
    assert today["date_cible"] == serie.index[-1] == seance_suivante(r.index[-1])
    for col in serie.columns.drop("date_prevision"):
        assert today[col] == derniere[col], col
