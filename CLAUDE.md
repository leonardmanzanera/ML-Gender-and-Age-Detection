# CLAUDE.md — Antigravity Vision

Lire ce fichier EN PREMIER à chaque session. Pour les sujets complexes,
les fichiers de référence détaillés sont dans `.claude/docs/`.

---

## Les 4 Piliers Fonctionnels

| Pilier | Pipeline | Moteur | Spécificité |
|---|---|---|---|
| **Analyse Biométrique Trackée** | V8 | `TrackedViTEngine` | YOLOv8 + ViT ONNX, tracking ID persistant, inférence async |
| **Sécurité / Reconnaissance** | V9 | `FaceRegistry` | Watchlist "Most Wanted", alertes rouges, match live |
| **Intelligence Esthétique** | V10 | `AestheticEngine` | Face Mesh 468pts, 5 scores géométriques, Radar Chart live |
| **Détection d'Environnement** | intégré | YOLOv8 COCO | Objets génériques, rectangles vert fluo |

---

## Structure du dépôt

```
/
├── CLAUDE.md             ← ce fichier
├── .claude/
│   └── docs/             ← documentation détaillée (lire selon le contexte)
│       ├── arch-conventions.md    # Architecture, thin pipeline, async, imports
│       ├── facemesh-aesthetic.md  # AestheticEngine, landmarks, scores, formules
│       ├── garmin-protocol.md     # Ingestion Garmin, nettoyage, analyse temporelle
│       └── ml-protocol.md        # Rigueur statistique ML, anti-leakage, séries
├── launcher.py           # Point d'entrée unique — alias `./start`
├── pipelines/            # v1_baseline.py … v10_beauty.py
├── src/
│   └── ag_vision/        # TOUTE la logique métier
│       ├── utils/
│       ├── aesthetic.py
│       ├── face_registry.py
│       └── engine_tracked.py
├── models/               # Poids binaires (.pt, .onnx, .task)
├── data/                 # watchlist/  best_shots/  logs/
├── config/               # YAML — toute configuration variable
└── requirements.txt      # macOS Apple Silicon M1/M2/M3 / arm64
```

---

## Règles absolues (à mémoriser)

### 1 — Thin Pipeline
Un fichier `pipelines/` : instanciation → boucle caméra → affichage. **C'est tout.**
Toute logique >10 lignes migre dans `src/ag_vision/`.
→ Détail complet : `.claude/docs/arch-conventions.md`

### 2 — 0-Latency (async >40ms)
**Rien ne bloque jamais `cv2.imshow()`.** Tout calcul IA >40ms = Thread + Queue obligatoire.
→ Pattern complet : `.claude/docs/arch-conventions.md`

### 3 — SOLID
Chaque moteur a une seule responsabilité. **Les moteurs ne dessinent jamais.**
Le Radar Chart, le masque Fibonacci et les alertes rouges = responsabilité du pipeline.
→ Détail : `.claude/docs/arch-conventions.md`

### 4 — Imports absolus uniquement
```python
from ag_vision.aesthetic import AestheticEngine   # ✅
from .aesthetic import AestheticEngine            # ❌
```

### 5 — Nommage PEP 8 strict
`PascalCase` classes · `snake_case` méthodes · `UPPER_SNAKE_CASE` constantes

---

## Contrat AestheticEngine (immuable)

Ne jamais casser cette structure — tous les pipelines en dépendent :

```python
analyze_face(landmarks) → {
    "radar":        { "phi", "symmetry", "regard", "harmonie", "teint" },  # 0–10
    "golden_score": float,   # utilisé par auto-capture pipeline V10
    "phi_pct":      float,
    "symmetry_pct": float,
    "raw_landmarks": list,   # 468 pts → masque Fibonacci (pipeline)
    "ratios":        dict,   # valeurs brutes debug
}
```

→ Formules complètes : `.claude/docs/facemesh-aesthetic.md`

---

## Style visuel Antigravity (dark theme)

```python
ANTIGRAVITY_STYLE = {
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9',       'grid.color': '#21262d',
}
COLORS = {
    'primary':  '#58a6ff',  # UI générale
    'recovery': '#3fb950',  # Objets COCO (vert fluo)
    'stress':   '#f85149',  # Alertes Watchlist (rouge)
    'neutral':  '#8b949e',
    'warning':  '#d29922',
}
```

---

## Quand lire quel fichier de référence

| Contexte | Fichier à lire |
|---|---|
| Nouveau pipeline, nouvelle classe, refacto | `.claude/docs/arch-conventions.md` |
| Modification de `aesthetic.py`, landmarks, scores | `.claude/docs/facemesh-aesthetic.md` |
| Données Garmin, FIT, GPX, HRV, sommeil | `.claude/docs/garmin-protocol.md` |
| ML, modélisation, dataset, features, séries temporelles | `.claude/docs/ml-protocol.md` |

---

## Checklist universelle

- [ ] Logique métier dans `src/ag_vision/` ?
- [ ] Classe à responsabilité unique ?
- [ ] Imports absolus `from ag_vision.xxx` ?
- [ ] Calcul >40ms → thread + Queue ?
- [ ] Pipeline thin (instanciation + boucle + affichage) ?
- [ ] Constantes landmarks en `UPPER_SNAKE_CASE` en tête de fichier ?
- [ ] Paramètres variables dans `config/` YAML ?
- [ ] Structure de `analyze_face()` préservée ?
- [ ] Auto-capture / Radar Chart / Fibonacci → pipeline, pas le moteur ?
