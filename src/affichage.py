"""Mise en forme lisible des nombres pour l'affichage (notebooks, application).

Ces fonctions ne modifient JAMAIS les données : elles renvoient une copie en
texte, réservée à l'affichage. Les CSV de ``reports/`` sont écrits à partir des
DataFrames d'origine et gardent leur précision complète.

Règles (écriture française, virgule décimale, jamais d'écriture scientifique) :
- p-value : « < 0,001 » si p < 0,001, sinon 3 décimales (0,027) ;
- AIC, BIC, log-vraisemblance, montants en USD : 0 décimale (3 502 ; 1 495) ;
- pourcentages (colonnes ``*_pct`` ou libellés contenant « % ») : 2 décimales ;
- colonnes entières (n, violations, retards…) : 0 décimale ;
- autres nombres : 3 décimales si |x| < 10, 2 si |x| < 1 000, 0 au-delà
  (dans un tableau, décimales communes si la colonne est homogène) ;
- milliers séparés par une espace fine insécable ; valeur manquante : « — ».
Le type de nombre est déduit du nom de la colonne (ou, pour une colonne de
valeurs mélangées, du libellé de la ligne).
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime

import numpy as np
import pandas as pd

SEUIL_P = 0.001
_MILLIERS = " "   # espace fine insécable
_MANQUANT = "—"

_RE_P = re.compile(r"(^p$|^p[_\- ]|p[_\- ]?value|pval|_p$)", re.IGNORECASE)
_RE_ZERO_DECIMALE = re.compile(r"^(aic|bic|loglik)$|usd|capital", re.IGNORECASE)
_RE_POURCENT = re.compile(r"_pct|%", re.IGNORECASE)


def _nombre(x: float, decimales: int) -> str:
    """Écrit x avec ``decimales`` décimales, virgule décimale et milliers séparés."""
    texte = f"{x:,.{decimales}f}"
    if float(texte.replace(",", "")) == 0:      # pas de « -0,000 »
        texte = texte.lstrip("-")
    return texte.replace(",", _MILLIERS).replace(".", ",")


def _est_manquant(x) -> bool:
    try:
        return x is None or x is pd.NaT or (isinstance(x, (float, np.floating)) and math.isnan(x))
    except TypeError:
        return False


def _decimales_generiques(ax: float) -> int:
    """Décimales d'un nombre « ordinaire » selon sa taille : 3 (< 10), 2 (< 1 000), 0 sinon."""
    if ax >= 1000:
        return 0
    if ax >= 10:
        return 2
    return 3


def formater_nombre(x, nom: str = "", entier: bool = False, decimales: int | None = None) -> str:
    """Formate une valeur pour l'affichage selon les règles du module.

    Paramètres
    ----------
    x         : valeur (nombre, booléen, date, texte).
    nom       : nom de la colonne ou libellé de la ligne ; sert à reconnaître
                une p-value, un AIC, un montant en USD ou un pourcentage.
    entier    : force 0 décimale (colonne de nombres entiers).
    decimales : décimales imposées pour un nombre « ordinaire » (même nombre de
                décimales sur toute une colonne) ; par défaut selon la taille de x.

    Retourne une chaîne ; le texte est renvoyé tel quel.
    """
    if _est_manquant(x):
        return _MANQUANT
    if isinstance(x, (bool, np.bool_)):
        return "oui" if x else "non"
    if isinstance(x, (pd.Timestamp, datetime, np.datetime64)):
        t = pd.Timestamp(x)
        return t.strftime("%Y-%m-%d") if t == t.normalize() else t.strftime("%Y-%m-%d %H:%M")
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, (int, np.integer)):
        return _nombre(float(x), 0)
    if not isinstance(x, (float, np.floating)):
        return str(x)

    x = float(x)
    if _RE_P.search(nom):
        return f"< {_nombre(SEUIL_P, 3)}" if x < SEUIL_P else _nombre(x, 3)
    if _RE_ZERO_DECIMALE.search(nom) or entier:
        return _nombre(x, 0)
    if _RE_POURCENT.search(nom):
        return _nombre(x, 2)
    return _nombre(x, decimales if decimales is not None else _decimales_generiques(abs(x)))


def texte_p(p: float) -> str:
    """Écrit une p-value dans une phrase : « p < 0,001 » ou « p = 0,027 »."""
    t = formater_nombre(p, "p_value")
    return "p " + t if t.startswith("<") else "p = " + t


def _colonne_entiere(s: pd.Series, nom: str) -> bool:
    """Vrai pour une colonne de flottants qui ne contient que des entiers (ex. retards avec NaN)."""
    if not pd.api.types.is_float_dtype(s) or _RE_P.search(nom) or _RE_POURCENT.search(nom):
        return False
    v = s.dropna().to_numpy()
    return v.size > 0 and bool(np.all(np.mod(v, 1) == 0))


def formater_tableau(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Copie TEXTE d'un tableau, avec des nombres lisibles (voir les règles du module).

    Les données d'origine ne sont pas modifiées : à utiliser uniquement pour
    l'affichage (``display(formater_tableau(df))``), jamais pour un export CSV.
    Dans une colonne numérique dont toutes les valeurs sont du même ordre de
    grandeur, toutes ont le même nombre de décimales ; si la colonne mélange
    des grandeurs (ex. persistance 0,973 et demi-vie 25,11), chaque valeur
    garde les décimales adaptées à sa taille.
    Pour une colonne de valeurs mélangées (dtype object), le libellé de la
    ligne sert à reconnaître le type de nombre (ex. ``var95_usd``).
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    for col in df.columns:
        s = df[col]
        nom = str(col)
        if s.dtype == object:
            out[col] = [formater_nombre(v, str(lig)) for lig, v in zip(df.index, s)]
            continue
        entier = _colonne_entiere(s, nom)
        decimales = None
        if pd.api.types.is_float_dtype(s) and s.notna().any():
            classes = {_decimales_generiques(abs(v)) for v in s.dropna()}
            if len(classes) == 1:          # colonne homogène : décimales communes
                decimales = classes.pop()
        out[col] = [formater_nombre(v, nom, entier, decimales) for v in s]
    return out
