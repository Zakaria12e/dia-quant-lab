"""Rendements log, cible décalée et découpage temporel.

Règle anti-fuite : la SEULE fonction de ce dépôt autorisée à regarder le futur
est ``make_target``. Elle est appelée une seule fois, dans ``build_frame``.
Aucun autre module ne réimplémente ce décalage.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "frozen.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def log_returns(closes: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Calcule les rendements log journaliers.

    r_t = ln(C_t / C_{t-1})

    C_t est connu à la clôture t — aucune information future n'est utilisée.
    La première ligne est NaN (pas de clôture précédente) et est conservée.

    Paramètres
    ----------
    closes : Series ou DataFrame de prix de clôture ajustés (index = date).

    Retourne un objet de même type avec les rendements log.
    """
    return np.log(closes / closes.shift(1))


def make_target(returns: pd.Series, horizon: int = 1) -> pd.Series:
    """Décale la série de rendements vers l'avant pour construire la cible.

    *** C'EST LA SEULE FONCTION DE CE DÉPÔT QUI REGARDE LE FUTUR. ***

    target[t] = returns[t + horizon]

    Autrement dit, la valeur en t est le rendement qu'on cherche à prévoir
    pour la séance t+horizon. La décision de la séance t est prise après
    la clôture de t, en n'utilisant que l'information disponible jusqu'à t.

    Paramètres
    ----------
    returns : Series de rendements log (index = date).
    horizon : nombre de séances à l'avance (défaut = 1, lu depuis frozen.yaml).

    Retourne une Series avec les mêmes index mais décalée de ``horizon``.
    Les ``horizon`` dernières valeurs sont NaN (futur inconnu).
    """
    return returns.shift(-horizon)


def build_frame(closes: pd.DataFrame) -> pd.DataFrame:
    """Construit le DataFrame prêt à l'entraînement.

    Colonnes :
    - ``dia_ret``  : r_t de DIA  — information connue à la clôture t
    - ``spy_ret``  : r_t de SPY  — information connue à la clôture t
    - ``target``   : r_{t+1} de DIA — cible à prévoir, construite via make_target

    Les lignes avec NaN (première ligne, et dernière après décalage) sont supprimées.

    Paramètres
    ----------
    closes : DataFrame format large avec colonnes DIA et SPY, index = date.

    Retourne un DataFrame trié par date.
    """
    cfg = _load_config()
    horizon = cfg["target"]["horizon"]

    ret = log_returns(closes)

    frame = pd.DataFrame({
        "dia_ret": ret["DIA"],
        "spy_ret": ret["SPY"],
        "target":  make_target(ret["DIA"], horizon=horizon),
    }, index=closes.index)

    return frame.dropna().sort_index()


def split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Découpe le DataFrame selon les fenêtres gelées dans frozen.yaml.

    Retourne un dictionnaire avec les clés ``train``, ``validation``, ``test``.

    Les fenêtres sont définies une seule fois dans configs/frozen.yaml et ne
    sont jamais recalculées à partir des données — ce qui garantit qu'aucune
    statistique future ne peut contaminer une fenêtre passée.
    """
    cfg = _load_config()
    windows = cfg["windows"]

    result = {}
    for name, (start, end) in windows.items():
        mask = (frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))
        result[name] = frame.loc[mask].copy()

    return result
