# 🧠 AG Vision (Antigravity) - Ultimate Edition

Une plateforme de vision par ordinateur avancée, optimisée pour **Mac M1 (Apple Silicon)**, combinant la détection d'objets, l'analyse d'âge/genre, la reconnaissance faciale, un coach de posture, et un mode confidentialité.

---

## 🚀 Fonctionnalités (Pipeline Ultimate)

L'application consolide plusieurs modèles de Machine Learning en temps réel :

1.  **Détection d'Objets** : YOLOv8 (80 classes COCO).
2.  **Analyse Démographique (Âge/Genre)** : Moteur asynchrone hybride.
    *   *Âge* : Vision Transformer (ViT) ONNX pour une haute précision (~4.5 ans MAE).
    *   *Genre* : Caffe CNN (résout les biais inhérents au ViT multiclasse).
3.  **🏷️ Face ID (Reconnaissance)** : Enregistrement de visages en "few-shot" via `face_recognition` (dlib) avec embeddings 128-dim.
4.  **🧘 Posture Coach** : Suivi du squelette via `mediapipe` (Tasks API) pour détecter si vous êtes voûté et recommander des pauses.
5.  **🛡️ Privacy Shield** : Floutage dynamique et automatique (GaussianBlur) des visages non reconnus par le Face ID.

---

## 🛠 Installation (Mac Apple Silicon)

Un script d'installation est fourni pour configurer l'environnement avec les bonnes versions compatibles M1 (notamment pour résoudre les conflits numpy/mediapipe/dlib).

### 1. Cloner ou télécharger le projet
```bash
git clone <votre_repo>
cd ml-gender-age-detection
```

### 2. Lancer le script d'installation
Ce script va créer un environnement virtuel (optionnel mais recommandé), installer dlib (qui nécessite la compilation), et installer toutes les dépendances.

```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

*(Ce script va configurer l'environnement virtuel, compiler `dlib`, et **télécharger automatiquement tous les modèles nécessaires** (~400 Mo)).*

---

## 🎮 Utilisation

Lancez le menu principal (Launcher Professionnel) :

```bash
python launcher.py
```

Choisissez l'option **`[7] 🧠 ULTIMATE`** pour lancer la pipeline complète.

### Contrôles Clavier (En direct)

| Touche | Action | Description |
| :---: | :--- | :--- |
| **`r`** | **Register Face** | Enregistre le visage actuellement à l'écran. *Attention : le focus bascule sur le terminal pour saisir le prénom.* |
| **`f`** | **Face ID Toggle** | Active/désactive la reconnaissance faciale. |
| **`p`** | **Privacy Toggle** | Floute automatiquement tout visage "Inconnu". |
| **`o`** | **Posture Toggle** | Active/désactive le squelette MediaPipe et les alertes dos voûté. |
| **`c`** | **Clear Database** | Supprime tous les visages enregistrés (demande confirmation dans le terminal). |
| **`q`** | **Quit** | Quitter l'application. |

---

## 📁 Architecture du Projet

```text
.
├── launcher.py               # Point d'entrée principal (Menu)
├── pipelines/                # Scripts d'exécution (V1 à V7 Ultimate)
│   ├── v_ultimate.py         # 🌟 La pipeline consolidée
│   └── ...
├── src/
│   └── ag_vision/            # Core Package Modulaire
│       ├── engine_async.py   # Moteur d'inférence asynchrone (ViT + Caffe)
│       ├── face_registry.py  # Base de données embeddings (dlib)
│       ├── posture_coach.py  # Détection de posture (MediaPipe)
│       ├── camera.py         # Capture optimisée M1
│       └── ...
├── models/                   # Poids des modèles (téléchargés automatiquement)
└── data/                     # Base de données logicielle (known_faces.pkl)
```

## ⚠️ Notes Techniques & Troubleshooting

- **Biais ViT** : Le modèle ViT officiel a un biais fort sur le logit de genre. Nous utilisons donc un moteur "Hybride" (Caffe pour le genre, ViT pour l'âge).
- **Conflits Python** : Assurez-vous d'utiliser `numpy < 2.0` (`1.26.4` recommandé) et `opencv-python == 4.10.0.84` pour éviter les crashs avec `mediapipe` et `dlib` sur Mac M1.
