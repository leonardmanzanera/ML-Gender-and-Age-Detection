---
name: antigravity-ml
description: >
  Protocole de rigueur statistique pour les projets de Machine Learning. Utilise ce skill
  dès que l'utilisateur parle de ML, modélisation, dataset, features, entraînement de modèle,
  séries temporelles, classification, régression, clustering, ou demande du code Python lié
  à la data science. S'applique aussi bien à l'exploration qu'à la production d'une architecture
  complète. Ne pas attendre que l'utilisateur mentionne "Antigravity" — déclencher ce skill
  dès qu'un contexte ML est détecté.
---

# Antigravity ML — Protocole de Rigueur Statistique

## Structure de réponse obligatoire

Toute réponse technique suit ce format en **3 temps** :

```
1. HYPOTHÈSE STATISTIQUE  → Justification théorique du choix
2. PLAN D'ACTION          → Étapes de prétraitement et modélisation
3. CODE                   → Implémentation après validation des étapes précédentes
```

Ne jamais sauter directement au code sans les étapes 1 et 2.

---

## Étape 0 — Détection du type de données

Avant tout, identifier la nature des données :

| Type détecté | Protocole applicable |
|---|---|
| Séries temporelles (index datetime, lag features) | → Section A |
| Tabulaire / classique | → Section B |
| Mixte ou inconnu | → Demander clarification |

---

## Section A — Séries Temporelles

### Checklist obligatoire

- [ ] **Test ADF** (Augmented Dickey-Fuller) pour la stationnarité
- [ ] Si non stationnaire → appliquer **FracDiff** avec `d ∈ [0, 1]`
- [ ] **Interdiction du K-Fold aléatoire** → utiliser exclusivement `TimeSeriesSplit`
- [ ] Vérifier l'absence de data leakage temporel (pas de features du futur)

```python
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit

# Test stationnarité
result = adfuller(series)
print(f"ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")

# Validation temporelle
tscv = TimeSeriesSplit(n_splits=5)
```

---

## Section B — Données Tabulaires

### Checklist obligatoire

- [ ] Graine globale fixée : `SEED = 42`
- [ ] Normalisation **locale par fold** (jamais globale avant split)
- [ ] Pipeline sklearn pour encapsuler le prétraitement

```python
SEED = 42
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])
scores = cross_val_score(pipe, X, y, cv=5, random_state=SEED)
```

---

## Audit Anti-Leakage (obligatoire avant tout entraînement)

Scanner systématiquement :

1. **Corrélation cible** : éliminer toute variable avec `corr(feature, target) > 0.90`
2. **Fuite de prétraitement** : normalisation dans le pipeline, pas avant le split
3. **Variables miroirs** : exclure IDs, timestamps bruts, index comme prédicteurs

```python
# Détection rapide de leakage
leaky = [col for col in X.columns if abs(X[col].corr(y)) > 0.90]
if leaky:
    print(f"⚠️  Variables suspectes : {leaky}")
```

---

## Architecture de projet (sortie standard)

Toujours proposer cette structure pour un projet complet :

```
project/
├── data/           # Ingestion et nettoyage
├── features/       # Feature engineering versionné
├── models/         # Définitions et poids
├── validation/     # Protocoles de test et métriques
└── config/         # Paramètres YAML (pas de hard-coding)
```

Et générer un `baseline.py` **avant** tout modèle complexe (LinearRegression ou RandomForest).

---

## Tracking et optimisation

- **MLflow ou W&B** : autologging sur chaque entraînement
- **Optuna** : tuning bayésien avec pruners pour les essais non prometteurs
- **Fonctions de perte** : évaluer l'asymétrie des risques → envisager Quantile Loss si erreurs asymétriques

```python
import optuna
import mlflow

def objective(trial):
    with mlflow.start_run(nested=True):
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        # ... entraînement
        mlflow.log_metric("val_score", score)
        return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---

## Diagnostics visuels obligatoires

Chaque phase de modélisation produit :

1. **Courbes loss/accuracy** → détection du surapprentissage
2. **Feature Importance** → interprétabilité métier
3. **Analyse des résidus** → validation des hypothèses du modèle

```python
import matplotlib.pyplot as plt

# Exemple résidus
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals); plt.axhline(0, color='red')
plt.title("Résidus vs Prédictions")
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30)
plt.title("Distribution des résidus")
plt.tight_layout()
```

---

## Règles de fin de session

- Supprimer les fonctions redondantes et le code mort
- Vérifier que aucun paramètre n'est hard-codé (tout dans `config/`)
- Privilégier les features interprétables (connaissance métier) avant la génération automatique massive
