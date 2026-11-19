"""Contrôles qualité OHLCV pour DIA et SPY."""

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


def audit(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Produit un tableau de contrôles qualité, une ligne par (contrôle, ticker).

    Colonnes du résultat : contrôle, ticker, nb_anomalies, détail.

    Paramètres
    ----------
    ohlcv : DataFrame format long (date, ticker, open, high, low, close, volume).
    """
    lignes = []

    tickers = sorted(ohlcv["ticker"].unique())

    # Calendrier NYSE pour détecter les séances manquantes
    nyse = mcal.get_calendar("NYSE")

    date_min = ohlcv["date"].min()
    date_max = ohlcv["date"].max()
    schedule = nyse.schedule(start_date=str(date_min.date()),
                             end_date=str(date_max.date()))
    seances_nyse = set(schedule.index.normalize())

    # ------------------------------------------------------------------ #
    # Contrôles par ticker                                                 #
    # ------------------------------------------------------------------ #
    for ticker in tickers:
        g = ohlcv[ohlcv["ticker"] == ticker].copy()
        dates_ticker = set(g["date"].dt.normalize())

        # 1. Valeurs manquantes par colonne
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in g.columns:
                continue
            n = int(g[col].isna().sum())
            lignes.append({
                "contrôle": f"valeurs_manquantes_{col}",
                "ticker": ticker,
                "nb_anomalies": n,
                "détail": f"{n} NaN" if n else "aucun",
            })

        # 2. Dates en double
        doublons = g["date"][g["date"].duplicated()].dt.date.tolist()
        lignes.append({
            "contrôle": "dates_en_double",
            "ticker": ticker,
            "nb_anomalies": len(doublons),
            "détail": str(doublons) if doublons else "aucun",
        })

        # 3. Prix ≤ 0
        for col in ["open", "high", "low", "close"]:
            if col not in g.columns:
                continue
            masque = g[col] <= 0
            n = int(masque.sum())
            lignes.append({
                "contrôle": f"prix_non_positif_{col}",
                "ticker": ticker,
                "nb_anomalies": n,
                "détail": str(g.loc[masque, "date"].dt.date.tolist()) if n else "aucun",
            })

        # 4. Volume négatif
        if "volume" in g.columns:
            masque = g["volume"] < 0
            n = int(masque.sum())
            lignes.append({
                "contrôle": "volume_negatif",
                "ticker": ticker,
                "nb_anomalies": n,
                "détail": str(g.loc[masque, "date"].dt.date.tolist()) if n else "aucun",
            })

        # 5. Volume nul
        if "volume" in g.columns:
            masque = g["volume"] == 0
            n = int(masque.sum())
            lignes.append({
                "contrôle": "volume_nul",
                "ticker": ticker,
                "nb_anomalies": n,
                "détail": str(g.loc[masque, "date"].dt.date.tolist()[:10]) if n else "aucun",
            })

        # 6. Incohérences OHLC
        if all(c in g.columns for c in ["open", "high", "low", "close"]):
            m1 = g["high"] < g[["open", "close"]].max(axis=1)
            m2 = g["low"] > g[["open", "close"]].min(axis=1)
            m3 = g["high"] < g["low"]
            for masque, label in [(m1, "high_inferieur_max_OC"),
                                   (m2, "low_superieur_min_OC"),
                                   (m3, "high_inferieur_low")]:
                n = int(masque.sum())
                lignes.append({
                    "contrôle": f"incoherence_ohlc_{label}",
                    "ticker": ticker,
                    "nb_anomalies": n,
                    "détail": str(g.loc[masque, "date"].dt.date.tolist()[:10]) if n else "aucun",
                })

        # 7. Séances manquantes vs calendrier NYSE
        manquantes = seances_nyse - dates_ticker
        # Exclure les dates hors de la plage du ticker
        manquantes = {d for d in manquantes
                      if date_min <= pd.Timestamp(d) <= date_max}
        n = len(manquantes)
        lignes.append({
            "contrôle": "seances_manquantes_nyse",
            "ticker": ticker,
            "nb_anomalies": n,
            "détail": (
                f"{n} séances (jours fériés US expliquent la majorité) — "
                f"ex: {sorted(manquantes)[:5]}"
            ) if n else "aucun",
        })

    # ------------------------------------------------------------------ #
    # Contrôles croisés DIA / SPY                                         #
    # ------------------------------------------------------------------ #
    if len(tickers) == 2:
        t1, t2 = tickers[0], tickers[1]
        d1 = set(ohlcv[ohlcv["ticker"] == t1]["date"].dt.normalize())
        d2 = set(ohlcv[ohlcv["ticker"] == t2]["date"].dt.normalize())

        seules_t1 = sorted(d1 - d2)
        seules_t2 = sorted(d2 - d1)

        lignes.append({
            "contrôle": f"seances_{t1}_absent_{t2}",
            "ticker": "croisé",
            "nb_anomalies": len(seules_t2),
            "détail": str([d.date() for d in seules_t2][:5]) if seules_t2 else "aucun",
        })
        lignes.append({
            "contrôle": f"seances_{t2}_absent_{t1}",
            "ticker": "croisé",
            "nb_anomalies": len(seules_t1),
            "détail": str([d.date() for d in seules_t1][:5]) if seules_t1 else "aucun",
        })

    # ------------------------------------------------------------------ #
    # Rendements extrêmes |r| > 5σ                                        #
    # ------------------------------------------------------------------ #
    for ticker in tickers:
        g = ohlcv[ohlcv["ticker"] == ticker].sort_values("date").copy()
        g["ret"] = np.log(g["close"] / g["close"].shift(1))
        sigma = g["ret"].std()
        masque = g["ret"].abs() > 5 * sigma
        extremes = g.loc[masque, ["date", "ret"]].copy()
        extremes["ret_pct"] = (extremes["ret"] * 100).round(2)
        n = len(extremes)
        detail = (
            extremes[["date", "ret_pct"]]
            .assign(date=extremes["date"].dt.date)
            .to_dict("records")
        ) if n else "aucun"
        lignes.append({
            "contrôle": "rendement_extreme_5sigma",
            "ticker": ticker,
            "nb_anomalies": n,
            "détail": (
                str(detail) + " — anomalies à expliquer (ex: mars 2020), pas à supprimer"
                if n else "aucun"
            ),
        })

    return pd.DataFrame(lignes)
