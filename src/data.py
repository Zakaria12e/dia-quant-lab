"""Collecte, snapshot et chargement des données OHLCV pour DIA et SPY."""

import os
from pathlib import Path
from datetime import date

import pandas as pd
import requests
import yfinance as yf
import yaml

# Racine du projet (deux niveaux au-dessus de ce fichier)
_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"
_CONFIG_PATH = _ROOT / "configs" / "frozen.yaml"

_COLONNES_OHLCV = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)



def download_snapshot(config: dict | None = None) -> Path:
    """Télécharge les données OHLCV ajustées de DIA et SPY, écrit un snapshot Parquet daté.

    Paramètres
    ----------
    config : dict facultatif — si None, lit configs/frozen.yaml.

    Retourne le chemin du fichier Parquet créé.

    Lève RuntimeError si DEMO_MODE=1 (téléchargement réseau interdit en mode démo).
    """
    if os.environ.get("DEMO_MODE", "0") == "1":
        raise RuntimeError(
            "DEMO_MODE=1 : le téléchargement réseau est interdit. "
            "Utilisez latest_snapshot() pour lire le snapshot local."
        )

    cfg = config or _load_config()
    tickers = [cfg["tickers"]["asset"], cfg["tickers"]["benchmark"]]
    start = cfg["download"]["start"]
    end = cfg["download"]["end"]

    # yfinance 1.8+ exige un cookie Yahoo valide. On ouvre d'abord la page
    # principale avec un User-Agent navigateur pour obtenir les cookies,
    # puis on passe la session à yf.download — sans ça on obtient HTTP 429.
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    session.get("https://finance.yahoo.com", timeout=15)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        session=session,
    )

    if raw.empty:
        raise RuntimeError(
            "Aucune donnée téléchargée. "
            "Vérifiez la connexion réseau ou réessayez dans quelques minutes."
        )

    morceaux = []
    for ticker in tickers:
        # Niveau 1 du MultiIndex = ticker ; niveau 0 = champ (Price/Ticker)
        # Selon la version de yfinance, le premier niveau peut s'appeler
        # "Price" ou directement le nom du champ.
        cols_ticker = [c for c in raw.columns if c[1] == ticker]
        if not cols_ticker:
            raise RuntimeError(f"Pas de données pour {ticker} dans la réponse yfinance.")
        piece = raw[cols_ticker].copy()
        piece.columns = [c[0].lower().replace(" ", "_") for c in piece.columns]
        piece.index.name = "date"

        # Coalescer close et adj_close (comportement variable selon les versions)
        if "close" not in piece.columns and "adj_close" in piece.columns:
            piece = piece.rename(columns={"adj_close": "close"})
        elif "close" in piece.columns and "adj_close" in piece.columns:
            piece["close"] = piece["close"].fillna(piece["adj_close"])
            piece = piece.drop(columns=["adj_close"])

        garder = [c for c in ["open", "high", "low", "close", "volume"]
                  if c in piece.columns]
        piece = piece[garder]
        piece["ticker"] = ticker
        morceaux.append(piece.reset_index())

    df = pd.concat(morceaux, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Nom du fichier : dia_spy_AAAA-MM-JJ.parquet
    today_str = date.today().isoformat()
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _DATA_DIR / f"dia_spy_{today_str}.parquet"
    df.to_parquet(path, index=False)

    print(f"Snapshot écrit : {path}  ({len(df)} lignes)")
    return path


def latest_snapshot() -> Path:
    """Renvoie le chemin du snapshot Parquet le plus récent dans data/.

    Lève FileNotFoundError s'il n'existe aucun snapshot.
    """
    snapshots = sorted(_DATA_DIR.glob("dia_spy_*.parquet"))
    if not snapshots:
        raise FileNotFoundError(
            f"Aucun snapshot trouvé dans {_DATA_DIR}. "
            "Lancez d'abord : python -c \"from src.data import download_snapshot; download_snapshot()\""
        )
    return snapshots[-1]


def load_ohlcv(path: Path | str | None = None) -> pd.DataFrame:
    """Charge le snapshot en format long (date, ticker, open, high, low, close, volume).

    Paramètres
    ----------
    path : chemin explicite ou None (utilise le snapshot le plus récent).
    """
    p = Path(path) if path is not None else latest_snapshot()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Rétrocompatibilité : les snapshots anciens peuvent avoir adj_close ou adj close
    for alias in ("adj_close", "adj close"):
        if alias in df.columns:
            if "close" not in df.columns or df["close"].isna().any():
                df["close"] = df.get("close", pd.Series(dtype=float)).fillna(df[alias])
            df = df.drop(columns=[alias])

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_closes(path: Path | str | None = None) -> pd.DataFrame:
    """Charge les clôtures ajustées en format large, séances communes uniquement.

    Colonnes : DIA, SPY. Index : date.

    Paramètres
    ----------
    path : chemin explicite ou None (utilise le snapshot le plus récent).
    """
    df = load_ohlcv(path)
    closes = (
        df.pivot(index="date", columns="ticker", values="close")
        .dropna(how="any")
        .sort_index()
    )
    closes.index = pd.to_datetime(closes.index)
    closes.index.name = "date"
    closes.columns.name = None
    return closes


if __name__ == "__main__":
    p = download_snapshot()
    closes = load_closes(p)
    print(closes.tail())
