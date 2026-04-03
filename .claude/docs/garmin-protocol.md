---
name: antigravity-garmin
description: >
  Protocole d'analyse rigoureux et exhaustif pour les exports massifs de données Garmin Connect.
  Déclencher ce skill dès que l'utilisateur mentionne Garmin, un export de montre connectée, des
  fichiers .fit / .gpx, des données d'activités sportives, de HRV, de sommeil, de VO2max, de charge
  d'entraînement, de stress ou de récupération. S'applique dès qu'un fichier CSV/JSON/FIT issu de
  Garmin Connect est présent, même partiellement. Couvre l'ingestion, le nettoyage, l'exploration,
  l'analyse avancée et la visualisation. Toujours utiliser ce skill AVANT de coder quoi que ce soit
  sur des données Garmin — il impose un protocole qualité non négociable.
---

# Antigravity Garmin — Protocole d'Analyse de Données

## Principe fondamental

> Les données Garmin sont des **séries temporelles biomédicales et sportives** à haute dimensionnalité.
> Elles ne s'analysent pas comme des données tabulaires classiques. Toute violation du protocole
> ci-dessous doit être signalée explicitement avant de continuer.

---

## ÉTAPE 0 — Inventaire des fichiers & détection du schéma

**Avant tout code**, effectuer un inventaire complet :

```python
import os, pathlib, pandas as pd

def inventory_garmin_export(root_dir: str) -> dict:
    """Inventaire structuré d'un export Garmin Connect."""
    inventory = {}
    for p in pathlib.Path(root_dir).rglob("*"):
        ext = p.suffix.lower()
        inventory.setdefault(ext, []).append(str(p))
    for ext, files in inventory.items():
        print(f"{ext:10s} → {len(files):5d} fichiers")
    return inventory
```

**Mapper les fichiers aux domaines de données → lire `references/garmin-schema.md`** pour la
correspondance complète fichier ↔ signification métier.

---

## ÉTAPE 1 — Règles d'ingestion par format

### 1A · Fichiers CSV (Garmin Connect export principal)

```python
import pandas as pd

def load_garmin_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Normalisation des colonnes
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(r'[\s\(\)/]', '_', regex=True)
                    .str.replace(r'_+', '_', regex=True)
                    .str.strip('_'))
    return df
```

**Checklist CSV obligatoire :**
- [ ] Détecter la colonne datetime principale (`date`, `start_time`, `timestamp`)
- [ ] Convertir en `pd.to_datetime()` avec `utc=True` puis localiser si besoin
- [ ] Identifier les unités (miles vs km, bpm, watts, metres) — voir `references/garmin-units.md`
- [ ] Repérer les valeurs sentinelles Garmin : `--`, `0.0` sur fréquence cardiaque, `999`, `-1`

### 1B · Fichiers FIT

```python
# pip install fitparse
from fitparse import FitFile

def fit_to_dataframe(filepath: str) -> dict[str, pd.DataFrame]:
    """Extrait toutes les message-types d'un fichier .fit en DataFrames séparés."""
    fit = FitFile(filepath)
    messages = {}
    for msg in fit.get_messages():
        name = msg.name
        row = {f.name: f.value for f in msg}
        messages.setdefault(name, []).append(row)
    return {k: pd.DataFrame(v) for k, v in messages.items()}
```

**Types de messages FIT à prioriser :** `record`, `session`, `lap`, `hrv`, `monitoring`

### 1C · Fichiers GPX

```python
import gpxpy

def gpx_to_dataframe(filepath: str) -> pd.DataFrame:
    with open(filepath) as f:
        gpx = gpxpy.parse(f)
    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                rows.append({
                    'lat': pt.latitude, 'lon': pt.longitude,
                    'elevation': pt.elevation, 'time': pt.time,
                    'speed': pt.speed
                })
    return pd.DataFrame(rows)
```

### 1D · Export JSON Garmin Connect ⭐ FORMAT PRIORITAIRE

Le JSON Garmin Connect est le format le plus riche mais aussi le plus complexe :
structure imbriquée profonde, timestamps en millisecondes, clés inconsistantes.

**→ Lire `references/json-parsing.md` OBLIGATOIREMENT avant tout parsing JSON.**

Ce fichier contient :
- La structure complète du dossier d'export JSON
- Les parsers spécialisés par type (activités, wellness, sommeil, HRV)
- Le pipeline `build_garmin_dataset()` — point d'entrée unique recommandé
- Les problèmes courants et leurs solutions

---

## ÉTAPE 2 — Nettoyage & Qualité des données

### Règle absolue : ne jamais imputer sans justification métier

| Situation | Action autorisée |
|---|---|
| HRV manquante (pas de mesure ce jour) | `NaN` → ne PAS imputer, flag `hrv_measured = False` |
| Fréquence cardiaque = 0 | Valeur sentinelle → remplacer par `NaN` |
| Distance < 10m sur activité > 5min | Probable GPS perdu → flag `gps_quality = 'low'` |
| Calories = 0 sur activité active | Donnée manquante → `NaN` |
| Sommeil < 1h ou > 16h | Outlier capteur → investiguer avant de supprimer |

```python
def clean_garmin_activities(df: pd.DataFrame) -> pd.DataFrame:
    # Valeurs sentinelles fréquentes
    sentinel_hr = df['avg_hr'].isin([0, 255, -1])
    df.loc[sentinel_hr, 'avg_hr'] = pd.NA

    # Distances impossibles
    if 'distance_km' in df.columns:
        df.loc[df['distance_km'] < 0, 'distance_km'] = pd.NA

    # Flag plutôt que suppression
    df['data_quality_flag'] = 'ok'
    df.loc[sentinel_hr, 'data_quality_flag'] = 'hr_sentinel'

    return df
```

### Audit de complétude obligatoire

```python
def audit_completeness(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    missing = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    report = missing[missing > 0].reset_index()
    report.columns = ['column', 'missing_pct']
    report['domain'] = domain
    print(f"\n📊 {domain} — {len(df)} lignes, {len(df.columns)} colonnes")
    print(report[report['missing_pct'] > 5].to_string(index=False))
    return report
```

---

## ÉTAPE 3 — Analyse par domaine de données

### ⚡ Lire le fichier de référence du domaine AVANT d'analyser

| Domaine détecté | Fichier de référence |
|---|---|
| **JSON Garmin (parsing)** | **`references/json-parsing.md` — à lire EN PREMIER** |
| Activités sportives | `references/domain-activities.md` |
| Santé & récupération (HRV, sommeil, stress) | `references/domain-health.md` |
| Physiologie (VO2max, charge, fitness) | `references/domain-physiology.md` |
| GPS & parcours | `references/domain-gps.md` |

---

## ÉTAPE 4 — Règles d'analyse temporelle (obligatoires)

Les données Garmin sont des **séries temporelles**. Appliquer systématiquement :

- [ ] **Index datetime** : toujours setter l'index sur la colonne temporelle principale
- [ ] **Resampling** explicite avec `rule` justifiée (`'1D'`, `'1W'`, `'4W'`)
- [ ] **Rolling windows** avec `min_periods` défini (pas de faux moyennes sur fenêtres incomplètes)
- [ ] **Jamais de corrélations brutes** entre métriques sans alignement temporel préalable
- [ ] **Détection des gaps** temporels avant toute interpolation

```python
def detect_temporal_gaps(df: pd.DataFrame, date_col: str,
                          expected_freq: str = '1D') -> pd.DataFrame:
    """Identifie les jours/périodes manquantes dans la série."""
    df = df.set_index(pd.to_datetime(df[date_col])).sort_index()
    full_range = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
    gaps = full_range.difference(df.index)
    if len(gaps) > 0:
        print(f"⚠️  {len(gaps)} gaps détectés sur {len(full_range)} jours attendus")
        print(f"   Premier gap : {gaps[0].date()}  |  Dernier gap : {gaps[-1].date()}")
    return df.reindex(full_range)
```

---

## ÉTAPE 5 — Visualisations standard

### Palette & style Antigravity

```python
import matplotlib.pyplot as plt
import seaborn as sns

ANTIGRAVITY_STYLE = {
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
}
plt.rcParams.update(ANTIGRAVITY_STYLE)

COLORS = {
    'primary': '#58a6ff',
    'recovery': '#3fb950',
    'stress': '#f85149',
    'neutral': '#8b949e',
    'warning': '#d29922',
}
```

### Visualisations par domaine → voir `references/viz-catalog.md`

---

## ÉTAPE 6 — Métriques clés calculées

### Performance

```python
def compute_training_load(df: pd.DataFrame,
                           atl_days: int = 7,
                           ctl_days: int = 42) -> pd.DataFrame:
    """
    ATL (Acute Training Load) = fatigue court terme
    CTL (Chronic Training Load) = forme long terme
    TSB (Training Stress Balance) = CTL - ATL
    """
    daily = df.resample('1D')['training_stress_score'].sum().fillna(0)
    df_load = daily.to_frame()
    df_load['ATL'] = daily.ewm(span=atl_days, min_periods=1).mean()
    df_load['CTL'] = daily.ewm(span=ctl_days, min_periods=1).mean()
    df_load['TSB'] = df_load['CTL'] - df_load['ATL']
    return df_load
```

### Récupération HRV

```python
def hrv_trend_analysis(hrv_series: pd.Series,
                        baseline_days: int = 60) -> pd.DataFrame:
    """
    Baseline HRV = moyenne mobile 60 jours
    Z-score = écart à la baseline en écarts-types
    Signal de récupération : z > 0.5 = bien récupéré, z < -1.5 = surcharge
    """
    baseline = hrv_series.rolling(baseline_days, min_periods=7).mean()
    std = hrv_series.rolling(baseline_days, min_periods=7).std()
    z_score = (hrv_series - baseline) / std
    return pd.DataFrame({
        'hrv': hrv_series,
        'hrv_baseline': baseline,
        'hrv_zscore': z_score,
        'recovery_signal': pd.cut(z_score,
            bins=[-999, -1.5, -0.5, 0.5, 999],
            labels=['surcharge', 'attention', 'neutre', 'récupéré'])
    })
```

---

## ÉTAPE 7 — Structure de sortie standard

Tout projet d'analyse Garmin produit :

```
garmin_analysis/
├── 00_inventory.py        # Inventaire + schéma détecté
├── 01_ingestion.py        # Loaders par format
├── 02_cleaning.py         # Nettoyage + audit qualité
├── 03_eda/                # Exploration par domaine
│   ├── activities_eda.ipynb
│   ├── health_eda.ipynb
│   └── physiology_eda.ipynb
├── 04_analysis/           # Métriques calculées
├── 05_viz/                # Visualisations exportées
├── data/
│   ├── raw/               # Fichiers originaux — JAMAIS modifiés
│   ├── processed/         # DataFrames nettoyés (.parquet)
│   └── features/          # Features calculées (.parquet)
└── config.yaml            # Chemins, paramètres (pas de hard-coding)
```

**Règle d'or : les fichiers `raw/` ne sont JAMAIS modifiés. Toujours travailler sur des copies.**

---

## Erreurs fréquentes à éviter absolument

| ❌ Erreur | ✅ Correction |
|---|---|
| Moyenner FC sur toute une activité sans retirer warm-up | Découper par `lap` ou par segment temporel |
| Comparer VO2max brut entre montres/algorithmes | Normaliser par même firmware ou traiter séparément |
| Analyser le sommeil sans vérifier le type de nuit (récup, voyage, maladie) | Ajouter colonne `sleep_context` |
| Corréler HRV et charge d'entraînement sans décalage temporel | Appliquer un lag de 24-48h sur HRV |
| Supprimer les activités sans GPS | Flaguer, pas supprimer (spinning, piscine, yoga) |
| Traiter pace et speed comme interchangeables | Pace = 1/speed — conversion explicite obligatoire |

---

## Checklist finale avant toute conclusion

- [ ] Toutes les unités sont documentées et cohérentes
- [ ] Les gaps temporels ont été détectés et documentés
- [ ] Les outliers sont flagués (pas supprimés sans justification)
- [ ] Les corrélations sont calculées avec alignement temporel correct
- [ ] Les visualisations incluent le nombre de points et la période analysée
- [ ] Les conclusions distinguent corrélation et causalité
- [ ] Les fichiers raw sont intacts
