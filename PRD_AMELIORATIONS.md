# PRD — AG Vision : Plan d'Améliorations Techniques
**Projet** : HybridFace / AG Vision  
**Auteur** : Léonard Manzanera  
**Date** : Avril 2026  
**Statut** : Draft v1.0

---

## Contexte

Ce document liste les améliorations techniques identifiées après relecture complète du code source (`src/ag_vision/`, `pipelines/`, `models/`). Les items sont classés par priorité : **P0** (critique / bug potentiel), **P1** (qualité, cohérence), **P2** (optimisation, refactoring). Chaque item inclut le problème constaté, la solution proposée et les critères de validation.

---

## Table des matières

1. [P0 — Validation landmarks manquante dans `AestheticEngine.analyze()`](#1)
2. [P0 — Midline de symétrie instable au roulis de tête](#2)
3. [P1 — `TemporalSmoother` : dead code à unifier](#3)
4. [P1 — Inconsistance des tailles de queue (4 vs 8)](#4)
5. [P1 — `age_net.caffemodel` : poids mort dans `models/`](#5)
6. [P1 — Normalisation du score Teint non calibrée](#6)
7. [P2 — Intégration du Posture Coach dans le rapport et la pipeline V10](#7)
8. [P2 — Privacy Shield : documenter et exposer comme feature officielle](#8)

---

## 1. Validation landmarks manquante dans `AestheticEngine.analyze()` {#1}

**Priorité** : P0 — Bug potentiel  
**Fichier** : `src/ag_vision/aesthetic.py`, méthode `analyze()`, ligne ~155

### Problème

```python
result = self.landmarker.detect(mp_image)
if not result.face_landmarks or len(result.face_landmarks) == 0:
    return None
landmarks = result.face_landmarks[0]  # ← accès direct sans vérification de longueur
```

MediaPipe Face Mesh retourne théoriquement 468 landmarks, mais sur un profil extrême (visage à 60°+ ou partiellement hors champ), la liste peut contenir moins d'éléments. L'accès direct à des indices comme `LM["chin_bottom"] = 152` provoquerait une `IndexError` non gérée, causant un crash silencieux du thread `AsyncAestheticEngine`.

### Solution proposée

Ajouter une vérification explicite avant tout accès aux landmarks :

```python
landmarks = result.face_landmarks[0]
if len(landmarks) < 468:
    return None  # profil trop extrême, skip silencieux
```

Optionnellement, logger un warning une fois (throttled) pour diagnostiquer les scènes problématiques :

```python
if len(landmarks) < 468:
    print(f"[AestheticEngine] Warning: expected 468 landmarks, got {len(landmarks)}. Skipping.")
    return None
```

### Critères d'acceptance

- `analyze()` ne lève jamais `IndexError` quelle que soit la pose faciale
- Le thread `AsyncAestheticEngine` reste stable sur une vidéo de 5 min avec profils variés
- Test unitaire : passer un crop de profil (>45°) → retourne `None` sans exception

---

## 2. Midline de symétrie instable au roulis de tête {#2}

**Priorité** : P0 — Précision scientifique  
**Fichier** : `src/ag_vision/aesthetic.py`, calcul de symétrie (section `Bilateral Symmetry`)

### Problème

```python
midline_x = (forehead[0] + chin[0]) / 2.0
```

La midline est calculée comme la moyenne des coordonnées X du front et du menton. Si la tête roule (inclinaison latérale), cette midline s'incline avec le visage. Un visage parfaitement symétrique mais légèrement penché recevra un score de symétrie dégradé, biaisant le résultat vers les sujets qui regardent droit face caméra.

**Illustration** : un tilt de 10° peut déplacer la midline de ~5-8 pixels sur un crop de 200px, soit une erreur de mesure de 3-5% sur les distances latérales.

### Solution proposée

Remplacer la midline par le **milieu du pont nasal** (landmark 168), qui est intrinsèquement centré sur l'axe de symétrie anatomique du visage indépendamment du roulis :

```python
# Avant
midline_x = (forehead[0] + chin[0]) / 2.0

# Après
nose_bridge = self._get_point(landmarks, "nose_bridge", w, h)  # landmark 168
midline_x = nose_bridge[0]
```

Ajouter le landmark dans le dictionnaire `LM` :

```python
LM = {
    ...
    "nose_bridge": 168,   # Point de référence médian robuste au roulis
}
```

**Alternative** (plus robuste) : calculer la midline comme moyenne de 3-5 points centraux (168, 6, 197, 195, 4) pour lisser l'effet de tout landmark bruité.

### Critères d'acceptance

- Le score de symétrie d'une même personne varie de moins de ±0,5 point entre 0° et ±15° de roulis
- Test : enregistrer le score symétrie d'un visage à 0°, +10°, -10° → écart-type < 0,3
- Aucune régression sur les scores en vue frontale

---

## 3. `TemporalSmoother` : dead code à unifier {#3}

**Priorité** : P1 — Dette technique  
**Fichiers** : `src/ag_vision/smoother.py`, `src/ag_vision/engine_tracked.py`

### Problème

`smoother.py` définit la classe `TemporalSmoother` avec `window_size=10`. Cette classe n'est importée dans aucun fichier du projet (confirmé par grep). Le `TrackedViTEngine` réimplémente la même logique de lissage en inline (`_smooth_age`, `_smooth_gender`) avec `SMOOTHING_WINDOW = 8`. Il existe donc **deux implémentations divergentes** :

| | `TemporalSmoother` | `TrackedViTEngine` inline |
|---|---|---|
| Fenêtre par défaut | 10 | 8 |
| Âge continu (ViT) | Moving Average | Moving Average |
| Genre (catégoriel) | Moving Mode + prob | Moving Mode + prob |
| Per-ID | Oui (defaultdict) | Oui (dict explicite) |
| Utilisé en V10 | **Non** | Oui |

### Solution proposée

**Option A (recommandée)** — Refactoriser `TrackedViTEngine` pour utiliser `TemporalSmoother` :

```python
# Dans engine_tracked.py
from ag_vision.smoother import TemporalSmoother

class TrackedViTEngine:
    SMOOTHING_WINDOW = 8

    def __init__(self, ...):
        self.smoother = TemporalSmoother(window_size=self.SMOOTHING_WINDOW)
        # Supprimer : self.age_history, self.gender_history
        # Supprimer : _smooth_age(), _smooth_gender()
```

Adapter `TemporalSmoother.update_and_get()` pour gérer le mode ViT (régression continue) déjà prévu via `is_regression=True`.

**Option B** — Supprimer `smoother.py` et documenter que le lissage est dans `engine_tracked.py`.

### Critères d'acceptance

- Une seule implémentation du lissage temporel dans le codebase
- `SMOOTHING_WINDOW` configurable en un seul endroit
- Tests : résultats identiques avant/après refactoring sur une séquence de 50 frames simulées

---

## 4. Inconsistance des tailles de queue (4 vs 8) {#4}

**Priorité** : P1 — Comportement non documenté  
**Fichiers** : `src/ag_vision/engine_tracked.py`, `src/ag_vision/engine_aesthetic.py`

### Problème

```python
# engine_tracked.py
MAX_QUEUE_SIZE = 8        # TrackedViTEngine

# engine_aesthetic.py
MAX_QUEUE_SIZE = 4        # AsyncAestheticEngine
```

En présence de 3 personnes dans le champ, chaque frame génère 3 soumissions. La queue esthétique (taille 4) se remplit en 1,3 frame contre 2,6 pour la queue ViT. L'`AsyncAestheticEngine` va donc silencieusement **éjecter des crops** (comportement `deque(maxlen=4)`), introduisant une latence plus élevée sur les scores esthétiques que sur les scores démographiques. Ce comportement asymétrique n'est nulle part documenté ni justifié.

### Solution proposée

**Option A** — Harmoniser à 8 (comportement cohérent) :
```python
# engine_aesthetic.py
MAX_QUEUE_SIZE = 8
```

**Option B** — Conserver 4 mais documenter explicitement la raison (l'analyse esthétique est ~80ms vs ~120ms pour le ViT, donc une queue plus petite peut être intentionnelle pour éviter de traiter des frames trop anciennes) :

```python
# engine_aesthetic.py
# Queue plus petite intentionnellement : l'analyse esthétique (~80ms) n'a pas besoin
# d'historique profond. Si la queue est pleine, le crop le plus ancien est éjecté —
# on préfère la fraîcheur à l'exhaustivité.
MAX_QUEUE_SIZE = 4
```

**Recommandation** : Option B avec documentation, car une queue plus petite sur le moteur le plus lent est une décision pertinente.

### Critères d'acceptance

- La valeur `MAX_QUEUE_SIZE` de chaque moteur est documentée avec sa justification
- Un commentaire dans le code explique le comportement d'éjection silencieux (`deque(maxlen=N)`)

---

## 5. `age_net.caffemodel` : poids mort dans `models/` {#5}

**Priorité** : P1 — Propreté du repo  
**Fichier** : `models/age_net.caffemodel`, `models/age_deploy.prototxt`

### Problème

Ces deux fichiers (~45 Mo selon le notebook) correspondent au modèle Caffe d'estimation d'âge par intervalles discrets utilisé dans V1/V2. Depuis V3, le ViT ONNX (`vit_age_gender.onnx`) assure la régression d'âge et le modèle Caffe n'est **jamais chargé** dans les pipelines V3+. Il alourdit inutilement le repo et crée une confusion sur l'architecture réelle du système.

### Solution proposée

```bash
# Retirer les fichiers du repo
git rm models/age_net.caffemodel
git rm models/age_deploy.prototxt
git commit -m "refactor: remove legacy Caffe age model (replaced by ViT in V3+)"
```

Si une compatibilité ascendante V1/V2 est souhaitée, déplacer dans `models/legacy/` avec un `README.md` explicatif.

Mettre à jour `scripts/install.sh` pour ne plus télécharger ces fichiers si c'est le cas.

### Critères d'acceptance

- `models/` ne contient que les modèles effectivement chargés par la pipeline V10
- `scripts/install.sh` ne télécharge pas `age_net.caffemodel` en mode V10
- Les pipelines V1/V2 (si conservées) documentent leur dépendance au modèle legacy

---

## 6. Normalisation du score Teint non calibrée {#6}

**Priorité** : P1 — Précision du scoring  
**Fichier** : `src/ag_vision/aesthetic.py`, section `Teint (Skin Texture)`

### Problème

```python
teint_score = max(0.0, 10.0 - (avg_var / 50.0))
```

Le facteur de normalisation `50.0` est arbitraire et très sensible aux conditions de capture :

- Webcam FaceTime 720p, bonne lumière → `avg_var` typiquement 20-80 → scores cohérents
- Webcam en basse lumière (gain ISO élevé) → `avg_var` peut atteindre 200-500 → `teint_score = 0` systématiquement
- Résolution de crop faible (visage loin) → `avg_var` proche de 0 → `teint_score = 10` artificiellement

Le score mesure actuellement plus la qualité de la capture vidéo que la qualité de la peau.

### Solution proposée

**Option A — Log-scaling** :
```python
# Variance log-normalisée, plus stable sur plusieurs ordres de grandeur
import math
log_var = math.log1p(avg_var)          # log(1 + var), toujours ≥ 0
# Calibration empirique : log(50) ≈ 3.9 → score 0; log(1) ≈ 0 → score 10
teint_score = max(0.0, 10.0 - (log_var / 0.45))
```

**Option B — Normalisation relative par frame** :
Calculer la variance sur un patch de référence non-facial (ex. fond uniforme ou épaule) et normaliser la variance faciale par rapport à elle. Trop complexe pour V10.

**Option C — Percentile adaptatif** :
Maintenir un historique glissant des variances et noter relativement au percentile courant. Pertinent pour une session longue.

**Recommandation** : Option A pour son implémentation simple et sa robustesse aux variations de matériel.

### Critères d'acceptance

- Sur une séquence avec éclairage variable (bon/mauvais), l'écart-type du score Teint pour une même personne est réduit d'au moins 30%
- La distribution des scores Teint sur 100 visages test couvre l'ensemble [3, 9] (pas de clustering aux extrêmes)
- Calibration documentée avec les valeurs de référence utilisées

---

## 7. Intégration du Posture Coach dans la pipeline V10 et le rapport {#7}

**Priorité** : P2 — Feature manquante dans le rapport  
**Fichier** : `src/ag_vision/posture_coach.py`, `pipelines/v10_beauty.py`

### Problème — côté rapport

Le `PostureCoach` est le 7ème modèle du système (MediaPipe Pose Landmarker) mais il est **absent du rapport**. La section 3 ne le mentionne pas, la section 1 ne le cite pas dans les applications. Pourtant, il est actif dans V10 via le toggle `[o]` et représente une fonctionnalité différenciante (feedback ergonomique temps réel).

La revendication "7 modèles hétérogènes issus de 4 familles architecturales" n'est justifiable que si le Posture Coach est documenté, car sans lui on compte 6 modèles (YOLOv8n, YOLOv8n-face, ViT, Caffe gender, dlib ResNet-34, MediaPipe Face Mesh).

### Problème — côté code

`PostureCoach` utilise `pose_landmarker_lite.task`, le modèle "lite" de MediaPipe Pose. Ce modèle n'est pas mentionné dans `requirements.txt` ni dans `scripts/install.sh`, ce qui peut causer un `FileNotFoundError` si l'utilisateur lance V10 sans avoir téléchargé le modèle.

### Solution proposée

**Côté code** :
1. Ajouter dans `scripts/install.sh` le téléchargement conditionnel de `pose_landmarker_lite.task`
2. Ajouter dans `posture_coach.py` un `_ensure_model()` similaire à celui de `AestheticEngine`

**Côté rapport** :
1. Ajouter une section 3.6 "Posture Coach et Privacy Shield" (ou fusionner avec 3.2)
2. Mettre à jour la section 1 : citer l'ergonomie/bien-être numérique comme cas d'usage
3. Mettre à jour le tableau section 5.1 pour inclure la latence du Pose Landmarker

### Critères d'acceptance

- V10 se lance sans erreur sur un environnement fresh install (sans `pose_landmarker_lite.task` pré-existant)
- Le rapport mentionne explicitement le Posture Coach avec son modèle et son fonctionnement
- La revendication "7 modèles" est justifiée par une liste exhaustive dans le rapport

---

## 8. Privacy Shield : documenter et exposer comme feature officielle {#8}

**Priorité** : P2 — Valorisation fonctionnelle  
**Fichier** : `pipelines/v10_beauty.py`, toggle `[p]`

### Problème

Le Privacy Shield (floutage `cv2.GaussianBlur` des visages "Inconnus") est une feature de protection des données pertinente dans le contexte RGPD/éthique ML, mais :
- Elle n'apparaît ni dans le rapport ni dans la section 3
- Elle n'est pas décrite dans le `README.md` de manière prominent
- Il n'existe pas de paramètre configurable pour l'intensité du flou (hardcodé)

### Solution proposée

**Côté code** :
1. Exposer le rayon du flou dans `config/` (paramètre `privacy_blur_radius`)
2. Ajouter un mode `[P2]` "anonymization complète" (floutage même des visages connus) pour les scénarios de démo publique

**Côté rapport** :
1. Mentionner le Privacy Shield dans la section 1 (applications : vie privée, conformité RGPD)
2. Ajouter une ligne dans le tableau de la section 5.2 comme "Scénario 4 — Privacy Shield"

### Critères d'acceptance

- L'intensité du flou est paramétrable via `config/` sans modifier le code
- La feature est décrite dans le rapport avec mention de son lien avec les enjeux éthiques/RGPD

---

## Synthèse et priorisation

| # | Item | Priorité | Effort | Impact |
|---|------|----------|--------|--------|
| 1 | Validation landmarks 468 | P0 | 30 min | Stabilité thread |
| 2 | Midline symétrie (roulis) | P0 | 2h | Précision scores |
| 3 | Unifier TemporalSmoother | P1 | 3h | Dette technique |
| 4 | Documenter queue size 4/8 | P1 | 30 min | Clarté code |
| 5 | Retirer age_net legacy | P1 | 30 min | Propreté repo |
| 6 | Calibrer score Teint | P1 | 2h | Précision scores |
| 7 | Posture Coach dans rapport | P2 | 4h | Complétude rapport |
| 8 | Privacy Shield feature | P2 | 3h | Valorisation |

**Effort total estimé** : ~15h de développement

**Ordre recommandé d'implémentation** :
1. Items P0 en priorité (1 + 2) — ~2h30
2. Nettoyage rapide (4 + 5) — ~1h
3. Items qualité (3 + 6) — ~5h
4. Items rapport (7 + 8) — ~7h

---

*Document généré lors de la revue de code du projet HybridFace — Avril 2026*
