"""Tests d'étanchéité temporelle — aucune information future ne doit remonter dans le passé."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import log_returns, make_target, build_frame, split
from src.models.econometrics import fit_state_space


# ---------------------------------------------------------------------------
# Fixtures : série synthétique propre
# ---------------------------------------------------------------------------

def _prix_synthetiques(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Génère des prix synthétiques DIA/SPY pour les tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    prix = pd.DataFrame({
        "DIA": 300.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
        "SPY": 350.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
    }, index=dates)
    prix.index.name = "date"
    return prix


# ---------------------------------------------------------------------------
# Tests de cohérence de make_target
# ---------------------------------------------------------------------------

def test_target_est_un_decalage_avant():
    """target[t] == dia_ret[t+1] pour tous les t (hors dernière ligne NaN)."""
    closes = _prix_synthetiques()
    ret = log_returns(closes)["DIA"]
    target = make_target(ret, horizon=1)

    # Vérification date par date sur la fenêtre intérieure
    for i in range(len(ret) - 1):
        t = ret.index[i]
        t1 = ret.index[i + 1]
        assert np.isclose(target[t], ret[t1], equal_nan=True), (
            f"target[{t.date()}] = {target[t]:.6f} "
            f"!= dia_ret[{t1.date()}] = {ret[t1]:.6f}"
        )


def test_target_derniere_valeur_nan():
    """La dernière valeur de target est NaN (futur inconnu)."""
    closes = _prix_synthetiques()
    ret = log_returns(closes)["DIA"]
    target = make_target(ret, horizon=1)
    assert np.isnan(target.iloc[-1]), "La dernière valeur de target devrait être NaN"


def test_build_frame_colonnes():
    """build_frame renvoie bien les colonnes dia_ret, spy_ret, target."""
    closes = _prix_synthetiques()
    frame = build_frame(closes)
    assert set(frame.columns) == {"dia_ret", "spy_ret", "target"}


def test_build_frame_pas_de_nan():
    """build_frame supprime toutes les lignes avec NaN."""
    closes = _prix_synthetiques()
    frame = build_frame(closes)
    assert frame.isna().sum().sum() == 0, "build_frame ne doit pas contenir de NaN"


def test_build_frame_target_coherent():
    """Dans le frame final, target[t] == dia_ret du lendemain."""
    closes = _prix_synthetiques()
    frame = build_frame(closes)
    # Pour chaque t sauf le dernier, target[t] = dia_ret[t+1]
    for i in range(len(frame) - 1):
        t  = frame.index[i]
        t1 = frame.index[i + 1]
        assert np.isclose(frame.loc[t, "target"], frame.loc[t1, "dia_ret"]), (
            f"target[{t.date()}] ne correspond pas à dia_ret[{t1.date()}]"
        )


def test_split_cles():
    """split() renvoie exactement les clés train / validation / test."""
    closes = _prix_synthetiques(n=200)
    frame = build_frame(closes)
    # Élargir les dates pour couvrir les fenêtres gelées
    dates = pd.date_range("2018-01-02", periods=200, freq="B")
    closes2 = _prix_synthetiques(n=200).copy()
    closes2.index = dates
    frame2 = build_frame(closes2)
    decoupage = split(frame2)
    assert set(decoupage.keys()) == {"train", "validation", "test"}


def test_split_pas_de_chevauchement():
    """Les fenêtres train / validation / test ne se chevauchent pas."""
    dates = pd.date_range("2018-01-02", periods=2300, freq="B")
    closes = _prix_synthetiques(n=2300)
    closes.index = dates
    frame = build_frame(closes)
    decoupage = split(frame)

    train_dates = set(decoupage["train"].index)
    val_dates   = set(decoupage["validation"].index)
    test_dates  = set(decoupage["test"].index)

    assert train_dates.isdisjoint(val_dates),  "Chevauchement train/validation"
    assert train_dates.isdisjoint(test_dates), "Chevauchement train/test"
    assert val_dates.isdisjoint(test_dates),   "Chevauchement validation/test"


def test_split_ordre_temporel():
    """train < validation < test (toutes les dates de train avant validation, etc.)."""
    dates = pd.date_range("2018-01-02", periods=2300, freq="B")
    closes = _prix_synthetiques(n=2300)
    closes.index = dates
    frame = build_frame(closes)
    decoupage = split(frame)

    if len(decoupage["train"]) and len(decoupage["validation"]):
        assert decoupage["train"].index.max() < decoupage["validation"].index.min()
    if len(decoupage["validation"]) and len(decoupage["test"]):
        assert decoupage["validation"].index.max() < decoupage["test"].index.min()


# ---------------------------------------------------------------------------
# Test anti-fuite SARIMAX — les paramètres gelés sur l'apprentissage
# ne doivent pas être contaminés par des données hors-échantillon
# ---------------------------------------------------------------------------

def _prix_synthetiques_longs(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Série suffisamment longue pour estimer un ARIMA(1,0,1)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    prix = pd.DataFrame({
        "DIA": 300.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
        "SPY": 350.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
    }, index=dates)
    prix.index.name = "date"
    return prix


def test_sarimax_gele_sur_apprentissage():
    """Les prévisions avant `cut` sont identiques même si les prix futurs changent.

    Protocole :
    1. Construire un frame de référence sur des prix propres.
    2. Dupliquer les prix et multiplier par 1,5 toutes les lignes à partir de `cut`
       (falsification des prix bruts, pas des rendements déjà calculés).
    3. Reconstruire le frame à partir des prix falsifiés.
    4. Estimer le même ARIMA(1,0,1) sur la fenêtre d'apprentissage dans les deux cas.
    5. Vérifier que :
       - les prévisions pour t ≤ cut sont identiques (atol=1e-10, rtol=1e-10),
       - au moins une prévision pour t > cut diffère.
    """
    # --- Données de référence ---
    closes_ref = _prix_synthetiques_longs(n=300)
    frame_ref  = build_frame(closes_ref)

    # Fenêtre d'apprentissage : 60 % des données
    n_train = int(len(frame_ref) * 0.6)
    train_mask_ref = pd.Series(False, index=frame_ref.index)
    train_mask_ref.iloc[:n_train] = True

    # Date de coupure : milieu de la zone hors-apprentissage
    cut = frame_ref.index[n_train + (len(frame_ref) - n_train) // 2]

    # --- Données falsifiées (prix bruts × 1,5 à partir de `cut`) ---
    # ⚠️ On falsifie les PRIX BRUTS, pas les rendements déjà calculés.
    closes_falsifies = closes_ref.copy()
    closes_falsifies.loc[closes_falsifies.index >= cut] *= 1.5
    frame_faux = build_frame(closes_falsifies)

    # Réaligner les masques sur les deux frames (index identique car même n)
    common_idx = frame_ref.index.intersection(frame_faux.index)
    frame_ref  = frame_ref.loc[common_idx]
    frame_faux = frame_faux.loc[common_idx]
    train_mask = pd.Series(False, index=common_idx)
    train_mask.iloc[:n_train] = True

    # --- Estimation ARIMA(1,0,1) sans exogène ---
    res_ref  = fit_state_space(frame_ref,  train_mask, order=(1, 0, 1),
                               seasonal_order=(0, 0, 0, 0), with_exog=False)
    res_faux = fit_state_space(frame_faux, train_mask, order=(1, 0, 1),
                               seasonal_order=(0, 0, 0, 0), with_exog=False)

    pred_ref  = res_ref.predictions
    pred_faux = res_faux.predictions

    # Les deux séries doivent partager le même index
    assert pred_ref.index.equals(pred_faux.index), (
        "Les prévisions de référence et falsifiées n'ont pas le même index."
    )

    # Prévisions avant `cut` : identiques (paramètres gelés sur le même apprentissage)
    avant_cut = pred_ref.index[pred_ref.index <= cut]
    assert len(avant_cut) > 0, "Aucune prévision avant `cut` — ajuster n ou cut."
    assert np.allclose(
        pred_ref.loc[avant_cut].values,
        pred_faux.loc[avant_cut].values,
        atol=1e-10, rtol=1e-10,
    ), (
        "Les prévisions avant `cut` diffèrent malgré des paramètres identiques — "
        "fuite temporelle potentielle."
    )

    # Prévisions après `cut` : doivent différer (les observations y ont changé)
    apres_cut = pred_ref.index[pred_ref.index > cut]
    assert len(apres_cut) > 0, "Aucune prévision après `cut` — ajuster n ou cut."
    assert not np.allclose(
        pred_ref.loc[apres_cut].values,
        pred_faux.loc[apres_cut].values,
        atol=1e-10, rtol=1e-10,
    ), (
        "Les prévisions après `cut` sont identiques malgré des observations différentes — "
        "les nouvelles observations ne semblent pas utilisées dans le filtre."
    )
