"""Tests d'étanchéité temporelle — aucune information future ne doit remonter dans le passé."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import log_returns, make_target, build_frame, split


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
