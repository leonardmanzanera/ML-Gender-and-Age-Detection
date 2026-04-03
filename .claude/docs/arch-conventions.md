---
name: antigravity-vision-arch
description: >
  Protocole d'architecture et de conventions de code pour le projet Antigravity Vision
  (détection genre/âge, tracking, analyse esthétique faciale). Déclencher ce skill dès
  que l'utilisateur demande d'ajouter une fonctionnalité, créer un nouveau pipeline,
  refactorer du code, ou écrire une nouvelle classe/moteur dans ce projet.
  Aussi applicable quand l'utilisateur mentionne ag_vision, AestheticEngine,
  TrackedViTEngine, FaceRegistry, un nouveau VX pipeline, ou tout fichier sous
  src/, pipelines/, ou scripts/. Ne jamais générer de code sans avoir appliqué
  ce protocole — structure, nommage et imports sont non négociables.
---

# Antigravity Vision — Architecture & Conventions

## Structure du Projet

```
/ (Racine)
├── pipelines/            # Scripts exécutables (v1_classic.py … v10_aesthetics.py)
├── src/
│   └── ag_vision/        # Package Python principal — toute l'intelligence métier
│       ├── utils/        # Caméra, téléchargement, config, helpers
│       ├── aesthetic.py  # AestheticEngine — analyse géométrique (Nombre d'Or, symétrie)
│       ├── face_registry.py  # FaceRegistry — base de données visages (.pkl / .jpg)
│       └── engine_tracked.py # TrackedViTEngine — tracking persistant + prédictions ViT
├── models/               # Poids binaires uniquement (.pt, .onnx, .task, .caffemodel)
├── data/                 # Runtime : Watchlist, Best Shots, Logs
├── config/               # Fichiers YAML de configuration
├── scripts/              # Installation et setup (setup.py, install.sh)
├── launcher.py           # Menu de lancement unifié — point d'entrée unique
└── requirements.txt      # Dépendances Mac M1 / arm64
```

## Règle Fondamentale : "Thin Pipeline"

Les fichiers dans `pipelines/` sont des **orchestrateurs légers**. Leur seul rôle :
1. Injecter `src` dans le path
2. Instancier les moteurs depuis `ag_vision`
3. Ouvrir la caméra et boucler sur les frames
4. Afficher le résultat

**Toute logique métier appartient à `src/ag_vision/`.** Si une fonction dépasse 10 lignes dans un pipeline, elle doit migrer dans le package.

### Template pipeline standard

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ag_vision.engine_tracked import TrackedViTEngine
from ag_vision.aesthetic import AestheticEngine
from ag_vision.utils.camera import open_camera

def main():
    engine = TrackedViTEngine()
    cam = open_camera()
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        results = engine.process(frame)
        # affichage uniquement ici
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
```

## Conventions de Nommage (PEP 8 strict)

| Élément | Convention | Exemples |
|---|---|---|
| Classes | `PascalCase` | `AestheticEngine`, `TrackedViTEngine`, `FaceRegistry` |
| Fonctions / Méthodes | `snake_case` | `analyze_face()`, `compute_phi_scores()` |
| Méthodes privées | `_snake_case` | `_get_point()`, `_load_model()` |
| Variables | `snake_case` | `face_crop`, `phi_score`, `track_id` |
| Constantes module | `UPPER_SNAKE_CASE` | `PHI = 1.618`, `LM_FOREHEAD_TOP = 10` |
| Fichiers pipeline | `v{N}_{nom_court}.py` | `v8_tracked.py`, `v10_aesthetics.py` |
| Fichiers moteur | `engine_{nom}.py` ou `{nom}.py` | `engine_tracked.py`, `aesthetic.py` |

### Principe S.O.L.I.D. appliqué
Chaque classe a **une seule responsabilité** :
- `AestheticEngine` → calculs géométriques uniquement (Nombre d'Or, symétrie, landmarks)
- `TrackedViTEngine` → tracking d'IDs + file d'attente ViT uniquement
- `FaceRegistry` → CRUD sur la base de données visages uniquement
- Les moteurs ne font **pas** d'affichage OpenCV — c'est le rôle du pipeline

## Conventions d'Imports

### Toujours : imports absolus depuis `ag_vision`

```python
# ✅ Correct
from ag_vision.aesthetic import AestheticEngine
from ag_vision.face_registry import FaceRegistry
from ag_vision.utils.camera import open_camera

# ❌ Jamais
from .aesthetic import AestheticEngine      # relatif
import aesthetic                            # sans package
```

### Injection de path (début de chaque fichier pipeline)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

### Ordre des imports dans un fichier

```python
# 1. Stdlib
import sys, threading, queue
from pathlib import Path

# 2. Path injection (pipelines uniquement)
sys.path.insert(0, ...)

# 3. Third-party
import cv2
import numpy as np
import onnxruntime as ort

# 4. ag_vision
from ag_vision.engine_tracked import TrackedViTEngine
```

## Pattern Asynchrone (modèles lourds)

Pour tout modèle dont l'inférence dépasse ~50ms (ViT, gros ONNX), appliquer ce pattern :

```python
import threading
import queue

class TrackedViTEngine:
    """Préfixe 'Tracked' ou 'Async' obligatoire pour les moteurs non-bloquants."""

    def __init__(self):
        self._prediction_queue = queue.Queue(maxsize=4)
        self._result_cache: dict = {}       # track_id → dernière prédiction
        self._worker = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._worker.start()

    def process(self, frame, detections):
        """Appel non-bloquant depuis la boucle caméra."""
        for det in detections:
            track_id = det["id"]
            if track_id not in self._result_cache:
                try:
                    self._prediction_queue.put_nowait((track_id, det["crop"]))
                except queue.Full:
                    pass
        return self._result_cache   # toujours retourner le cache actuel

    def _inference_loop(self):
        """Thread de fond — traite la file sans bloquer la caméra."""
        while True:
            track_id, crop = self._prediction_queue.get()
            prediction = self._run_model(crop)   # inférence lourde ici
            self._result_cache[track_id] = prediction
```

**Règle** : La boucle caméra ne doit jamais attendre une inférence. Si le résultat n'est pas prêt, on affiche le cache ou rien.

## Constantes et Configuration

Les constantes de landmarks MediaPipe Face Mesh vivent **en tête du fichier** qui les utilise, pas dans un fichier de config :

```python
# aesthetic.py — en haut du fichier
PHI = 1.618033988749895

# Landmarks faciaux (indices MediaPipe Face Mesh 468 points)
LM_FOREHEAD_TOP   = 10
LM_CHIN_BOTTOM    = 152
LM_LEFT_EYE_OUTER = 33
LM_RIGHT_EYE_OUTER = 263
LM_NOSE_TIP       = 4
LM_LEFT_MOUTH     = 61
LM_RIGHT_MOUTH    = 291
```

Les paramètres variables (seuils, chemins, résolution caméra) vont dans `config/` en YAML et sont chargés via `ag_vision/utils/config.py`.

## Checklist avant de générer du code

Avant d'écrire la moindre ligne pour ce projet, vérifier :

- [ ] La logique métier va-t-elle dans `src/ag_vision/` (pas dans le pipeline) ?
- [ ] La classe respecte-t-elle le principe de responsabilité unique ?
- [ ] Le nommage suit-il PascalCase / snake_case / UPPER_SNAKE_CASE ?
- [ ] Les imports sont-ils absolus (`from ag_vision.xxx import YYY`) ?
- [ ] Si le modèle est lourd (>50ms), utilise-t-on un thread + Queue ?
- [ ] Le pipeline reste-il "thin" (instanciation + boucle + affichage uniquement) ?
- [ ] Les nouvelles constantes de landmarks sont-elles déclarées en UPPER_SNAKE_CASE ?

## Ajout d'une nouvelle version (V{N})

1. Créer `pipelines/v{N}_{nom}.py` — thin pipeline uniquement
2. Si nouvelle logique métier → nouveau fichier dans `src/ag_vision/`
3. Ajouter l'entrée dans `launcher.py`
4. Documenter les dépendances si nouvelles dans `requirements.txt`

Ne jamais copier-coller de logique d'un ancien pipeline vers un nouveau. Factoriser dans `src/`.
