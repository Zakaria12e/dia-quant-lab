"""Tests de base pour src/data.py — chargement du snapshot et intégrité des données."""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Garantit que les imports de src/ fonctionnent depuis le dossier tests/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import latest_snapshot, load_ohlcv, load_closes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ohlcv():
    return load_ohlcv()


@pytest.fixture(scope="module")
def closes():
    return load_closes()


# ---------------------------------------------------------------------------
# Tests du snapshot
# ---------------------------------------------------------------------------

def test_snapshot_existe():
    """Le snapshot le plus récent existe et est lisible."""
    path = latest_snapshot()
    assert path.exists(), f"Snapshot introuvable : {path}"


def test_deux_tickers_presents(ohlcv):
    """DIA et SPY sont tous deux présents dans le snapshot."""
    tickers = set(ohlcv["ticker"].unique())
    assert "DIA" in tickers, "DIA absent du snapshot"
    assert "SPY" in tickers, "SPY absent du snapshot"


def test_colonnes_attendues(ohlcv):
    """Le format long contient les colonnes obligatoires."""
    attendues = {"date", "ticker", "open", "high", "low", "close", "volume"}
    manquantes = attendues - set(ohlcv.columns)
    assert not manquantes, f"Colonnes manquantes : {manquantes}"


def test_index_trie_sans_doublons(ohlcv):
    """Pour chaque ticker, les dates sont triées et sans doublons."""
    for ticker, groupe in ohlcv.groupby("ticker"):
        dates = groupe["date"].reset_index(drop=True)
        assert dates.is_monotonic_increasing, f"Dates non triées pour {ticker}"
        assert dates.nunique() == len(dates), f"Doublons de dates pour {ticker}"


def test_fermetures_format_large(closes):
    """load_closes renvoie un DataFrame avec colonnes DIA et SPY, index date."""
    assert "DIA" in closes.columns
    assert "SPY" in closes.columns
    assert closes.index.name == "date"
    assert closes.index.is_monotonic_increasing
    assert closes.isna().sum().sum() == 0, "Des NaN subsistent dans load_closes"


def test_seances_communes_uniquement(ohlcv, closes):
    """Les séances du format large sont exactement les séances communes DIA ∩ SPY."""
    dates_dia = set(ohlcv[ohlcv["ticker"] == "DIA"]["date"])
    dates_spy = set(ohlcv[ohlcv["ticker"] == "SPY"]["date"])
    communes = dates_dia & dates_spy
    assert set(closes.index) == communes


def test_prix_positifs(ohlcv):
    """Les prix de clôture sont strictement positifs."""
    assert (ohlcv["close"] > 0).all(), "Des prix de clôture nuls ou négatifs détectés"


def test_demo_mode_bloque_telechargement(monkeypatch):
    """DEMO_MODE=1 interdit le téléchargement et lève RuntimeError."""
    monkeypatch.setenv("DEMO_MODE", "1")
    from src.data import download_snapshot
    with pytest.raises(RuntimeError, match="DEMO_MODE"):
        download_snapshot()
