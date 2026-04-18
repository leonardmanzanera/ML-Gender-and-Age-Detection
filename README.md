# 🧠 AG Vision — HybridFace System

**Projet académique de vision par ordinateur — YNOV / Léonard Manzanera — Avril 2026**

Système d'analyse faciale en temps réel combinant **7 modèles hétérogènes** issus de **4 familles architecturales** (CNN, Transformer, Graph Neural Network, Détecteurs single-shot) au sein d'une pipeline unifiée asynchrone.

---

## 📋 Table des matières

1. [Fonctionnalités](#-fonctionnalités)
2. [Modes disponibles](#-modes-disponibles-via-le-launcher)
3. [Installation rapide](#-installation-rapide)
4. [Installation manuelle](#-installation-manuelle-étape-par-étape)
5. [Utilisation](#-utilisation)
6. [Architecture du projet](#-architecture-du-projet)
7. [Modèles utilisés](#-modèles-utilisés)
8. [Notes techniques](#-notes-techniques)

---

## ✨ Fonctionnalités

| Module | Modèle | Famille |
|--------|--------|---------|
| Détection de visages | YOLOv8n-Face + BoT-SORT | Single-Shot Detector |
| Détection d'objets (80 classes) | YOLOv8n (COCO) | Single-Shot Detector |
| Estimation d'âge (régression ±4.5 ans MAE) | Vision Transformer (ViT) ONNX | Transformer |
| Classification de genre | Caffe CNN (AlexNet) | CNN |
| Reconnaissance faciale (Face ID) | dlib ResNet-34 (embeddings 128D) | CNN |
| Analyse esthétique (Nombre d'Or, Symétrie) | MediaPipe Face Mesh (468 landmarks) | GNN |
| Coach de posture (détection voûtement) | MediaPipe Pose Landmarker | GNN |

**Architecture clé** : Pattern asynchrone (Thread + Queue) qui délègue les inférences lourdes (~80–120ms) à des workers parallèles, permettant au flux vidéo de rester fluide sans freeze.

---

## 🎬 Modes disponibles (via le Launcher)

| # | Version | Description |
|---|---------|-------------|
| 1 | V1 Baseline | Détection SSD + Caffe CNN (pipeline V1 originale) |
| 2 | V2 YOLOv8 + Caffe | YOLOv8 face detection + Caffe âge/genre |
| 3 | V3 ViT Synchrone | YOLOv8 + ViT ONNX (synchrone — démontre le bottleneck) |
| 4 | **V3.1 ViT Async** | YOLOv8 + ViT ONNX asynchrone (architecture optimale) |
| 5 | V4 Objets | Détection d'objets COCO uniquement |
| 6 | V5 Unified | Objets + ViT Async (combiné) |
| 7 | **V6 Ultimate** | Objets + ViT + Face ID + Posture + Privacy |
| 8 | **V8 Tracked** | Multi-personnes avec IDs isolés (recommandé pour démo) |
| 9 | V9 Watchlist | Reconnaissance + alerte en cas de correspondance |
| 10 | **V10 Aesthetics** | Analyse de beauté (Nombre d'Or + Symétrie + Radar Chart) |

---

## ⚡ Installation rapide

### Prérequis

- **Python 3.9, 3.10 ou 3.11** (Python 3.12 peut poser problème avec dlib)
- **Webcam** (interne ou USB)
- **macOS, Ubuntu, ou Windows** avec au minimum 8 Go de RAM

### macOS (recommandé — script automatique)

```bash
# 1. Autoriser l'exécution du script
chmod +x scripts/install.sh

# 2. Lancer l'installation (environ 5-10 minutes)
./scripts/install.sh
```

> Le script installe cmake, crée un environnement virtuel, installe toutes les dépendances, et **télécharge automatiquement tous les modèles ML** (~450 Mo).

---

## 🛠 Installation manuelle (étape par étape)

Si le script automatique ne fonctionne pas, suivez ces étapes :

### Étape 1 — Installer CMake (nécessaire pour dlib)

```bash
# macOS
brew install cmake

# Ubuntu / Debian
sudo apt-get install cmake build-essential

# Windows
# Télécharger depuis https://cmake.org/download/
```

### Étape 2 — Créer un environnement virtuel (recommandé)

```bash
python3 -m venv ag_env
source ag_env/bin/activate   # macOS / Linux
# ag_env\Scripts\activate    # Windows
```

### Étape 3 — Installer les dépendances Python

```bash
pip install --upgrade pip cmake
pip install -r requirements.txt
```

> ⚠️ **dlib** peut prendre 3 à 5 minutes à compiler. C'est normal.

### Étape 4 — Télécharger les modèles ML

```bash
python3 scripts/setup.py
```

> Télécharge automatiquement : YOLOv8, ViT ONNX, Caffe CNN, MediaPipe Tasks, dlib face predictor... (~450 Mo total)

---

## 🎮 Utilisation

### Lancer le menu principal

```bash
# Si vous utilisez l'environnement virtuel :
source ag_env/bin/activate

# Lancer le launcher interactif :
python3 launcher.py

# Ou via le raccourci shell :
./start
```

### Contrôles clavier (dans toutes les fenêtres)

| Touche | Action |
|--------|--------|
| `q` | Quitter |
| `f` | Toggle Face ID (reconnaître les visages enregistrés) |
| `p` | Toggle Privacy Shield (flouter les visages inconnus) |
| `o` | Toggle Posture Coach (squelette + alertes dos voûté) |
| `m` | Toggle Masque d'Or (visualisation des 468 landmarks) |
| `r` | Enregistrer un visage (saisir le prénom dans le terminal) |
| `c` | Effacer la base de données des visages |

---

## 📁 Architecture du projet

```
AG Vision/
│
├── launcher.py                    ← Point d'entrée (menu interactif)
├── start                          ← Raccourci shell (./start)
├── requirements.txt               ← Dépendances Python
│
├── pipelines/                     ← Scripts d'exécution des versions
│   ├── v1_baseline.py             ← V1: SSD + Caffe CNN
│   ├── v2_yolo_caffe.py           ← V2: YOLOv8 + Caffe CNN
│   ├── v3_vit_onnx.py             ← V3: ViT synchrone (baseline synchrone)
│   ├── v3_1_vit_async.py          ← V3.1: ViT asynchrone (optimal)
│   ├── v4_object_detection.py     ← V4: Détection objets COCO
│   ├── v5_unified_vision.py       ← V5: Objets + ViT
│   ├── v6_ultimate.py             ← V6: Pipeline complète
│   ├── v8_tracked.py              ← V8: Multi-personnes trackées
│   ├── v9_watchlist.py            ← V9: Reconnaissanse + alerte
│   └── v10_beauty.py              ← V10: Analyse esthétique Nombre d'Or
│
├── src/ag_vision/                 ← Package Python principal
│   ├── aesthetic.py               ← Calcul beauté (Phi, symétrie, 468 landmarks)
│   ├── engine_async.py            ← Moteur async ViT+Caffe — Thread+Queue (V3.1–V6)
│   ├── engine_tracked.py          ← Moteur multi-personnes avec IDs isolés (V8+)
│   ├── engine_aesthetic.py        ← Wrapper async AestheticEngine — Thread+Queue (V10)
│   ├── face_registry.py           ← Base de données embeddings 128D (dlib)
│   ├── posture_coach.py           ← Détection posture (MediaPipe Pose)
│   ├── smoother.py                ← Lissage temporel (Moving Average/Mode)
│   ├── watchlist.py               ← Gestion liste de reconnaissance
│   └── utils/                     ← Utilitaires (config, caméra, download)
│
├── models/                        ← Poids des modèles ML (auto-téléchargés)
│   ├── yolov8n.pt
│   ├── yolov8n-face.pt
│   ├── vit_age_gender.onnx        ← Vision Transformer (âge, 344 Mo)
│   ├── gender_net.caffemodel      ← Caffe CNN genre
│   ├── gender_deploy.prototxt
│   ├── face_landmarker.task       ← MediaPipe 468 landmarks
│   ├── pose_landmarker_lite.task  ← MediaPipe pose 33 points
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── legacy/                    ← Caffe âge (V1/V2 uniquement)
│
├── config/
│   └── params.yaml                ← Configuration centralisée
│
├── data/
│   └── known_faces.pkl            ← Base de données des visages enregistrés
│
├── figures/                       ← Figures (architecture, landmarks, benchmarks)
│   ├── fig1_architecture.png
│   ├── fig2_landmarks_phi.png
│   ├── fig3_radar_chart.png
│   └── chronogram_v3_vs_v31.png  ← Comparatif sync/async frame-by-frame
│
└── scripts/
    ├── install.sh                 ← Script d'installation automatique (macOS)
    ├── setup.py                   ← Téléchargement des modèles ML (~450 Mo)
    └── chronogram_v3_vs_v31.py   ← Génération du chronogramme de performance
```

---

## 🤖 Modèles utilisés

| Modèle | Fichier | Taille | Usage |
|--------|---------|--------|-------|
| YOLOv8n (COCO) | `yolov8n.pt` | ~6 Mo | Détection 80 classes |
| YOLOv8n-Face | `yolov8n-face.pt` | ~6 Mo | Détection + tracking visages |
| ViT ONNX | `vit_age_gender.onnx` | ~345 Mo | Régression d'âge |
| Caffe Gender CNN | `gender_net.caffemodel` | ~45 Mo | Classification genre |
| MediaPipe Face Mesh | `face_landmarker.task` | ~3.5 Mo | 468 landmarks 3D |
| MediaPipe Pose | `pose_landmarker_lite.task` | ~5.6 Mo | 33 landmarks corporels |
| SSD Face Detector | `res10_300x300_ssd_iter_140000.caffemodel` | ~10 Mo | Détecteur SSD (V1) |

---

## ⚠️ Notes techniques

### Compatibilité M1 / Apple Silicon
Le projet est optimisé pour **Mac M1/M2/M3**. YOLO est configuré avec `device="mps"` et `imgsz=320` dans V3.1 pour exploiter le GPU Metal (8× plus rapide que CPU). Sur machine non-Apple, remplacer `device="mps"` par `device="cpu"` dans `pipelines/v3_1_vit_async.py`.

### Conflits de dépendances connus
| Problème | Solution |
|----------|----------|
| `mediapipe` crash avec NumPy 2.x | Utiliser `numpy==1.26.4` (< 2.0) |
| `dlib` ne compile pas | Vérifier que `cmake` est installé AVANT `pip install` |
| Conflit `cv2` / `cv2.contrib` | Ne pas installer les deux indépendamment — utiliser les versions du `requirements.txt` |

### Performance attendue (Mac M1, 1 personne, webcam 720p)
| Pipeline | FPS attendus | Description |
|----------|-------------|-------------|
| V1 (baseline) | 15–20 | SSD + Caffe, traitement CPU |
| V3 (synchrone) | 3–5 | ViT bloquant le thread principal (démonstation du bottleneck) |
| V3.1 (async, sans fix) | 9–12 | ViT en thread parallèle, YOLO encore bloquant |
| V3.1 (async + MPS fix) | 25–45 | YOLO sur Metal GPU + imgsz=320 |
| V8/V9 (tracked) | 12–20 | Multi-personnes + isolation des IDs |
| V10 (complet) | 6–10 | 2×YOLO + ViT + MediaPipe simultanés |

> Les FPS affichés représentent la **cadence du thread d'affichage**. Les inférences lourdes (ViT ~120ms, aesthetic ~80ms) s'exécutent en parallèle sur des workers asynchrones dédiés.

---

*Projet académique — AG Vision HybridFace — Léonard Manzanera — Avril 2026*
