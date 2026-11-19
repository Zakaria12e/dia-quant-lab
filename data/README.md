# Données — DIA & SPY · Provenance et dictionnaire

## Provenance

| Champ | Valeur |
|:---|:---|
| Source | Yahoo Finance via `yfinance` 1.8.0 |
| Instruments | **DIA** (SPDR Dow Jones Industrial Average ETF Trust) et **SPY** (SPDR S&P 500 ETF Trust) |
| Marché | NYSEArca — fuseau horaire America/New_York, clôture 16h00 ET |
| Fréquence | Journalière (1 ligne = 1 séance de bourse) |
| Période téléchargée | 2018-01-01 → 2026-10-31 |
| Convention d'ajustement | `auto_adjust=True` — les prix de clôture, ouverture, haut et bas sont tous corrigés des dividendes et des fractionnements d'actions (splits). La colonne `close` correspond donc à l'ancien « Adj Close ». |
| Date d'acquisition | 2026-11-19 |
| Fichier snapshot | `data/dia_spy_2026-11-19.parquet` |
| Format | Apache Parquet, format long (une ligne par séance et par ticker) |

---

## Dictionnaire des colonnes

| Colonne | Type | Unité | Description |
|:---|:---|:---|:---|
| `date` | datetime64[ms] | — | Date de la séance boursière (minuit UTC, horaire America/New_York) |
| `ticker` | string | — | Symbole boursier : `DIA` ou `SPY` |
| `open` | float64 | USD | Prix d'ouverture ajusté |
| `high` | float64 | USD | Prix le plus haut de la séance, ajusté |
| `low` | float64 | USD | Prix le plus bas de la séance, ajusté |
| `close` | float64 | USD | Prix de clôture ajusté (dividendes et splits corrigés) |
| `volume` | float64 | Nombre d'actions | Volume échangé sur la séance (non ajusté) |

---

## Statistiques générales

| Fenêtre | Période | Séances DIA & SPY |
|:---|:---|:---|
| Apprentissage | 2018-01-02 → 2022-12-30 | 1 259 |
| Validation | 2023-01-03 → 2024-12-31 | 502 |
| Test | 2025-01-02 → 2026-10-30 | 459 |
| **Total commun** | 2018-01-02 → 2026-10-30 | **2 220** |

---

## Lacunes et anomalies constatées (audit `src/quality.py`)

Résultat de `audit(load_ohlcv())` sur le snapshot du 2026-11-19 :

### Valeurs manquantes
Aucune valeur manquante dans les colonnes OHLCV pour DIA ou SPY.

### Dates en double
Aucune.

### Prix non positifs
Aucun.

### Volume nul
Aucun.

### Incohérence OHLC
- **SPY — 2018-01-11 (1 séance)** : `high < max(open, close)` détecté.
  Probablement un artefact d'ajustement rétroactif de dividende par Yahoo Finance.
  Décision : **conserver** (l'artefact est mineur, le prix ajusté reste économiquement cohérent).

### Séances manquantes vs calendrier NYSE
Les seules séances absentes sont les **jours fériés américains** (Thanksgiving, Noël, Martin Luther King Jr. Day, etc.) qui ne sont pas des jours de bourse — comportement normal.

### Rendements extrêmes |r| > 5σ — à expliquer, pas à supprimer

**DIA — 11 séances :**

| Date | Rendement (%) | Contexte |
|:---|---:|:---|
| 2020-03-09 | −8,10 | Crash COVID + guerre des prix pétroliers (Arabie/Russie) |
| 2020-03-11 | −6,02 | Déclaration pandémie OMS |
| 2020-03-12 | −10,60 | Choc de panique, circuit-breaker déclenché |
| 2020-03-13 | +9,01 | Rebond après annonce de mesures fédérales |
| 2020-03-16 | −13,65 | Pire séance depuis 1987 (Black Monday 2.0) |
| 2020-03-18 | −6,82 | Poursuite de la panique |
| 2020-03-24 | +10,45 | Rebond après annonce plan de relance 2 000 Md$ |
| 2020-03-26 | +5,95 | Continuation du rebond |
| 2020-04-06 | +7,29 | Espoir de pic épidémique |
| 2020-06-11 | −7,06 | Crainte de deuxième vague COVID |
| 2025-04-09 | +7,57 | Annonce de pause dans les droits de douane (guerre commerciale) |

**SPY — 8 séances :**

| Date | Rendement (%) | Contexte |
|:---|---:|:---|
| 2020-03-09 | −8,13 | Crash COVID |
| 2020-03-12 | −10,06 | Choc de panique |
| 2020-03-13 | +8,20 | Rebond fédéral |
| 2020-03-16 | −11,59 | Pire séance depuis 1987 |
| 2020-03-24 | +8,67 | Annonce plan de relance |
| 2020-04-06 | +6,50 | Espoir de pic |
| 2025-04-04 | −6,03 | Escalade guerre commerciale |
| 2025-04-09 | +9,99 | Pause droits de douane |

**Décision :** toutes ces anomalies correspondent à des événements macroéconomiques documentés.
Elles sont **conservées** dans le dataset : les supprimer biaiserait les estimations de volatilité et masquerait une information utile sur le comportement des modèles en période de crise.

---

## Comment recharger les données

**Télécharger un nouveau snapshot (connexion réseau requise) :**
```bash
python -c "from src.data import download_snapshot; download_snapshot()"
```

**Charger le snapshot existant sans réseau (`DEMO_MODE=1`) :**
```bash
set DEMO_MODE=1
python -c "from src.data import load_closes; print(load_closes().shape)"
```

**En Python :**
```python
from src.data import load_ohlcv, load_closes

ohlcv  = load_ohlcv()          # format long : date, ticker, open, high, low, close, volume
closes = load_closes()          # format large : index=date, colonnes=DIA/SPY, séances communes
```
