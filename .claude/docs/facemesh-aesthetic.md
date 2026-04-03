---
name: antigravity-facemesh
description: >
  Protocole complet pour travailler sur l'AestheticEngine d'Antigravity Vision :
  calculs géométriques sur les 468 landmarks MediaPipe Face Mesh, scores Phi /
  symétrie / harmonie / regard / teint, et structure du dictionnaire retourné.
  Déclencher ce skill dès que l'utilisateur mentionne AestheticEngine, aesthetic.py,
  Nombre d'Or, score Phi, symétrie bilatérale, canthal tilt, golden_score, radar chart,
  masque de Fibonacci, ou toute modification des calculs faciaux. Ne jamais réécrire
  ou étendre ces formules sans avoir appliqué ce protocole.
---

# Antigravity Vision — AestheticEngine & Face Mesh

## Rôle de la classe

`AestheticEngine` (`src/ag_vision/aesthetic.py`) est le seul responsable des calculs
géométriques faciaux. Elle reçoit les 468 landmarks MediaPipe Face Mesh d'un visage
cropé et retourne un dictionnaire structuré de scores. Elle ne fait ni détection, ni
affichage, ni tracking.

## Contrat de `analyze_face()`

```python
def analyze_face(self, landmarks: list) -> dict:
    """
    Args:
        landmarks: liste de 468 points (x, y, z normalisés) issus de MediaPipe Face Mesh

    Returns:
        dict avec les clés suivantes (voir détail ci-dessous)
    """
```

### Structure complète du dictionnaire retourné

```python
{
    # ── Scores Radar Chart (normalisés 0.0 → 10.0) ──────────────────────────
    "radar": {
        "phi":       float,  # Proportions Nombre d'Or (5 sous-ratios moyennés)
        "symmetry":  float,  # Symétrie bilatérale (7 paires de points clés)
        "regard":    float,  # Canthal tilt + ouverture pupillaire
        "harmonie":  float,  # Équilibre des 3 tiers faciaux (Léonard de Vinci)
        "teint":     float,  # Texture peau via variance laplacienne
    },

    # ── Scores globaux ───────────────────────────────────────────────────────
    "golden_score":   float,  # Score composite final (0–10)
                              # = 50% phi + 50% symmetry + bonus regard/harmonie/teint
    "phi_pct":        float,  # Score Phi en % pur (0–100)
    "symmetry_pct":   float,  # Score symétrie en % pur (0–100)

    # ── Données de rendu & debug ─────────────────────────────────────────────
    "raw_landmarks":  list,   # Les 468 points bruts (pour le masque de Fibonacci)
    "ratios": {               # Valeurs mathématiques exactes (transparence algo)
        "face_h_w":    float, # Hauteur visage / largeur pommettes  (idéal: 1.618)
        "eye_iod":     float, # Largeur yeux / distance inter-oculaire  (idéal: 1.0)
        "nose_lip":    float, # Distance nez-lèvre / lèvre-menton  (idéal: 1.618)
        "nose_mouth":  float, # Largeur nez / largeur bouche  (idéal: 0.618)
        "thirds":      float, # Front-sourcils / sourcils-nez  (idéal: 1.0)
    },
}
```

## Calcul du Score Phi (détail)

Le score Phi est la **moyenne de 5 sous-ratios**, chacun comparé à sa valeur idéale.

| Sous-ratio | Formule | Idéal | Landmarks impliqués |
|---|---|---|---|
| Rapport global | hauteur_visage / largeur_pommettes | **1.618** | `LM_FOREHEAD_TOP`, `LM_CHIN_BOTTOM`, pommettes G/D |
| Harmonie horizontale | largeur_yeux / distance_inter-oculaire | **1.0** | coins ext/int des deux yeux |
| Proportions verticales | dist_nez_lèvre / dist_lèvre_menton | **1.618** | `LM_NOSE_TIP`, lèvre sup, `LM_CHIN_BOTTOM` |
| Largeur relative | largeur_nez / largeur_bouche | **0.618** | ailes du nez, coins bouche |
| Tiers faciaux | dist_front_sourcils / dist_sourcils_nez | **1.0** | `LM_FOREHEAD_TOP`, sourcils, `LM_NOSE_TIP` |

### Normalisation vers 0–10

```python
PHI = 1.618033988749895

def _ratio_to_score(self, measured: float, ideal: float) -> float:
    """Convertit un ratio mesuré en score 0–10 par proximité à l'idéal."""
    deviation = abs(measured - ideal) / ideal   # déviation relative
    score = max(0.0, 1.0 - deviation) * 10.0
    return round(score, 2)
```

## Calcul de la Symétrie Bilatérale

Moyenne sur **7 paires de points clés** (distance gauche vs distance miroir droit) :

```python
SYMMETRY_PAIRS = [
    # (landmark_gauche, landmark_droit, description)
    (33,  263, "coins externes des yeux"),
    (133, 362, "coins internes des yeux"),
    (61,  291, "coins de la bouche"),
    (234, 454, "pommettes"),
    (127, 356, "mâchoire latérale"),
    (21,  251, "arcades sourcilières"),
    (93,  323, "bord inférieur de la mâchoire"),
]

def _compute_symmetry(self, lm: list) -> float:
    """Score de symétrie 0–10 : 10 = parfaitement symétrique."""
    scores = []
    mid_x = lm[1][0]  # axe central approximatif (landmark 1 = entre les yeux)
    for lm_left, lm_right, _ in SYMMETRY_PAIRS:
        dist_left  = abs(lm[lm_left][0]  - mid_x)
        dist_right = abs(lm[lm_right][0] - mid_x)
        ratio = min(dist_left, dist_right) / max(dist_left, dist_right + 1e-6)
        scores.append(ratio * 10.0)
    return round(sum(scores) / len(scores), 2)
```

## Calcul du Canthal Tilt (score Regard)

```python
def _compute_canthal_tilt(self, lm: list) -> float:
    """
    Angle d'inclinaison de l'axe oculaire par rapport à l'horizontal.
    Positif = yeux en amande (hunter eyes), négatif = yeux tombants.
    Retourne un score 0–10 centré sur l'idéal esthétique (~+3° à +7°).
    """
    left_outer  = lm[33]   # coin externe œil gauche
    left_inner  = lm[133]  # coin interne œil gauche
    right_inner = lm[362]  # coin interne œil droit
    right_outer = lm[263]  # coin externe œil droit

    # Angle de la ligne externe gauche → externe droit
    dx = right_outer[0] - left_outer[0]
    dy = right_outer[1] - left_outer[1]
    angle_deg = math.degrees(math.atan2(-dy, dx))  # y inversé (coords image)
    # Normalisation : idéal ≈ +3°, score 10 si dans la plage [2°, 8°]
    ...
```

## Calcul des 3 Tiers (Harmonie)

```python
# Les 3 tiers faciaux (Léonard de Vinci)
# Tiers 1 : hairline (LM_FOREHEAD_TOP=10)  → sourcils (LM_BROW=70)
# Tiers 2 : sourcils (LM_BROW=70)          → base du nez (LM_NOSE_BASE=2)
# Tiers 3 : base du nez (LM_NOSE_BASE=2)   → menton (LM_CHIN_BOTTOM=152)
# Score 10 si les 3 tiers sont égaux (déviation < 5%)
```

## Constantes de Landmarks

```python
PHI = 1.618033988749895

# Points structurants (MediaPipe Face Mesh 468 points)
LM_FOREHEAD_TOP    = 10
LM_CHIN_BOTTOM     = 152
LM_NOSE_TIP        = 4
LM_NOSE_BASE       = 2
LM_LEFT_EYE_OUTER  = 33
LM_LEFT_EYE_INNER  = 133
LM_RIGHT_EYE_INNER = 362
LM_RIGHT_EYE_OUTER = 263
LM_LEFT_MOUTH      = 61
LM_RIGHT_MOUTH     = 291
LM_LEFT_BROW       = 70
LM_RIGHT_BROW      = 300
```

## Calcul du Teint (Variance Laplacienne)

```python
def _compute_skin_score(self, face_crop: np.ndarray) -> float:
    """
    Évalue la finesse du grain de peau par variance laplacienne sur le crop visage.
    Haute variance = texture marquée (cicatrices, pores visibles) → score bas.
    Basse variance  = peau lisse → score haut.
    Seuils empiriques : variance < 80 → score ~9–10, variance > 400 → score ~2–3.
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalisation inverse : score = 10 * exp(-variance / SKIN_VARIANCE_SCALE)
    SKIN_VARIANCE_SCALE = 150.0
    score = 10.0 * math.exp(-variance / SKIN_VARIANCE_SCALE)
    return round(min(10.0, score), 2)
```

## Formule du Golden Score

```python
def _compute_golden_score(self, radar: dict) -> float:
    """
    Score composite final (0–10).
    Base : 50% Phi + 50% Symétrie.
    Bonus : +0.5 max si Regard > 8.0, +0.3 max si Harmonie > 8.0, +0.2 max si Teint > 8.0.
    """
    base = 0.5 * radar["phi"] + 0.5 * radar["symmetry"]
    bonus = 0.0
    if radar["regard"]   > 8.0: bonus += 0.5 * (radar["regard"]   - 8.0) / 2.0
    if radar["harmonie"] > 8.0: bonus += 0.3 * (radar["harmonie"] - 8.0) / 2.0
    if radar["teint"]    > 8.0: bonus += 0.2 * (radar["teint"]    - 8.0) / 2.0
    return round(min(10.0, base + bonus), 2)
```

## Checklist avant de modifier `aesthetic.py`

- [ ] Toute nouvelle métrique suit la normalisation 0–10 via `_ratio_to_score()`
- [ ] Les nouveaux landmarks sont déclarés comme constantes `UPPER_SNAKE_CASE` en haut du fichier
- [ ] La structure du dictionnaire retourné par `analyze_face()` est préservée (rétro-compatibilité pipelines)
- [ ] Un nouveau sous-score optionnel va dans `radar` — pas à la racine du dict
- [ ] `raw_landmarks` et `ratios` sont toujours présents (nécessaires au rendu du masque Fibonacci et au debug)
- [ ] `golden_score` reste calculé en dernier, après tous les scores `radar`
