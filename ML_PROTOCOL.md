# Antigravity ML Core Protocol: Manifeste de Rigueur Statistique

Ce document définit les règles impératives et le protocole opérationnel pour l'agent Antigravity. Chaque action entreprise dans un projet de Machine Learning doit se conformer strictement à ces directives pour garantir des résultats de grade professionnel.

## 1. Protocole de Réflexion et d'Autonomie

Antigravity ne doit jamais générer de code sans avoir préalablement validé sa logique statistique via une chaîne de pensée (Chain of Thought).

**Structure de Réponse Obligatoire :** Toute proposition technique doit suivre un format en trois temps :
1.  **Hypothèse Statistique :** Justification théorique du choix (ex: pourquoi ce biais inductif est adapté).
2.  **Plan d'Action :** Détail des étapes de prétraitement et de modélisation.
3.  **Code :** Mise en œuvre finale uniquement après validation des étapes précédentes.

**Audit d'Autocritique :** Avant chaque exécution, l'agent doit scanner le script pour détecter :
*   **Data Leakage :** Identifier toute variable ayant une corrélation supérieure à 0.90 avec la cible.
*   **Fuite de Prétraitement :** Vérifier que la normalisation est locale par fold et non globale.
*   **Variables Miroirs :** Éliminer les identifiants ou timestamps utilisés par erreur comme prédicteurs.

## 2. Polyvalence et Rigueur Statistique

L'agent doit adapter dynamiquement son approche selon la nature des données détectées.

### Données de Séries Temporelles :
*   **Test de Stationnarité :** Exécuter systématiquement un test d'Augmented Dickey-Fuller (ADF).
*   **Préservation de la Mémoire :** Si non stationnaire, appliquer une différenciation fractionnaire (FracDiff) avec un ordre $d \in [0, 1]$ pour conserver un maximum de corrélation avec la série originale.
*   **Validation Spécifique :** Interdiction stricte du K-Fold aléatoire. Utiliser exclusivement `TimeSeriesSplit` avec des fenêtres glissantes ou extensibles.

### Données Tabulaires / Classiques :
*   **Validation Croisée :** Utiliser une validation croisée standard avec une graine d'aléatoirité fixée globalement à `SEED = 42` pour assurer la reproductibilité.

### Optimisation Mathématique :
*   **Fonctions de Perte :** Évaluer l'asymétrie des risques métier. Si nécessaire, implémenter une "Quantile Loss" ou une perte personnalisée pour pénaliser plus lourdement les erreurs critiques (ex: sous-estimation du risque).
*   **Optimiseurs :** Justifier le choix de l'optimiseur (Adam, SGD, etc.) selon la topologie de la surface de perte et la densité des données.

## 3. Standards de Développement (Baseline & MLOps)

L'efficacité d'Antigravity repose sur une structure modulaire et un suivi rigoureux.

*   **Architecture du Projet :** Le répertoire doit obligatoirement suivre cette segmentation :
    *   `data/` : Ingestion et nettoyage.
    *   `features/` : Logique de feature engineering versionnée.
    *   `models/` : Définitions et poids des architectures.
    *   `validation/` : Protocoles de test et métriques.
    *   `config/` : Paramètres YAML (aucun hard-coding autorisé).
*   **Protocole de Baseline :** Créer systématiquement un script `baseline.py` (modèle naïf type Régression Linéaire ou Forêt Aléatoire) avant toute complexification pour valider le flux de données et servir de point de référence.
*   **Tracking et Tuning :**
    *   **MLOps :** Intégrer des hooks d'autologging via MLflow ou Weights & Biases pour chaque entraînement.
    *   **Optimisation :** Privilégier le tuning bayésien avec Optuna en définissant des espaces de recherche cohérents et des "pruners" pour arrêter les essais non prometteurs.

## 4. Diagnostic Visuel et Validation

Le code généré doit inclure des outils de vérification visuelle pour confirmer l'intuition statistique.

*   **Graphiques Obligatoires :** Chaque phase de modélisation doit produire :
    *   Courbes de perte (Loss) et de précision pour détecter le surapprentissage.
    *   Importance des caractéristiques (Feature Importance) pour l'interprétabilité métier.
    *   Analyse des résidus pour valider les hypothèses du modèle.
*   **Validation des Pipelines :** Utiliser systématiquement `sklearn.pipeline` pour encapsuler le prétraitement et éviter l'oubli du ré-ajustement des scalers sur chaque fold.

## 5. Hygiène et Sécurité du Code

Pour prévenir les hallucinations et la dégradation du projet, Antigravity suit ces règles de maintenance.

*   **Anti-Hallucination :** Utiliser la commande `@docs` pour pointer vers la documentation officielle des bibliothèques (Scikit-learn, PyTorch) et éviter l'invention de paramètres.
*   **Refactorisation :** À la fin de chaque session, supprimer les fonctions redondantes et le code mort ("sédimentation") pour maintenir un contexte global clair.
*   **Contrôle de la Complexité :** Favoriser des caractéristiques interprétables basées sur la connaissance métier avant de recourir à la génération automatique massive.
