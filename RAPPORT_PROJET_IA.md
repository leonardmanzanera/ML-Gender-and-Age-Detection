# 🎓 Rapport de Projet : Ingénierie de l'IA et Vision par Ordinateur (AG Vision)

## 1. Introduction & Objectifs du Projet

Le projet **AG Vision (Super-Vision)** s'inscrit dans la conception d'une plateforme d'Intelligence Artificielle de pointe, spécialisée dans la vision par ordinateur multi-modale en temps réel (Edge AI). 

L'objectif principal était de dépasser les simples algorithmes de détection pour concevoir une architecture logicielle robuste capable de consolider **cinq modèles hétérogènes d'IA** tournant simultanément sur une machine Edge (Apple Silicon M1/ARM64) sans dégradation des performances.

Ce projet répond à plusieurs enjeux techniques critiques :
-   **Détection d'Objets & Visages (YOLOv8)** : Vitesse et précision.
-   **Analyse Démographique (ViT / CNN)** : Extraction de métadonnées (Âge/Genre) avec lissage temporel.
-   **Reconnaissance Faciale (Few-Shot Learning)** : Identification via embeddings vectoriels.
-   **Estimation de Pose (MediaPipe)** : Détection d'anomalies posturales en temps réel.
-   **Filtrage Dynamique (Privacy Shield)** : Traitement d'images conditionnel basé sur la reconnaissance faciale.

## 2. Parcours Architectural (De la Baseline à la "Super-Vision")

Le développement a suivi une méthodologie itérative (CI/CD algorithmique) pour isoler la complexité :

### Phase 1 : Baseline (Single-Shot Detectors)
La première itération reposait sur des architectures classiques (SSD et ResNet-10 via le framework Caffe) pour valider le pipeline bas niveau de capture vidéo (via `cv2.VideoCapture`). Bien que stable, la précision de la détection de visages sous-performait sur les angles inclinés.

### Phase 2 : Modernisation (YOLOv8 et ONNX Runtime)
Pour réduire l'empreinte mémoire et augmenter le *Recall*, la détection a été migrée vers l'architecture `YOLOv8-Face` et `YOLOv8-COCO` (Ultralytics). Cette transition a imposé le passage à l'environnement `ONNX Runtime` pour standardiser l'exécution des *computational graphs* sur processeur ARM.

### Phase 3 : "Ultimate Pipeline" (Consolidation Multi-Modèle)
La version finale intègre une architecture asynchrone modulaire fusionnant 5 modèles dans un seul flux vidéo avec une latence maintenue sous les 60ms par frame (environ 15-20 FPS stabilisés).

## 3. Défis Techniques & Résolutions Avancées

C'est dans l'intégration système et l'orchestration des modèles que résident les véritables réussites d'ingénierie de ce projet.

### A. Compatibilité ARM64 et Environnement C++ (M1 Bottleneck)
- **Le Problème** : L'écosystème Python (Numpy 2.x) a introduit des ruptures d'API C ABI critiques sur Mac M1, créant des *Segmentation Faults* (Core Dumped) lors de l'appel aux bibliothèques pré-compilées C++ telles que `dlib` (reconnaissance faciale) et `mediapipe` (MediaPipe Tasks API).
- **La Solution** : Audit rigoriste des graphes de dépendances et rétrogradation (downgrading) explicite de l'environnement : `numpy==1.26.4` et `opencv-python==4.10.0.84`. Développement d'un script d'automatisation d'environnement Bash (`install.sh`) pour forcer la compilation locale de `dlib` via `CMake`.

### B. Inférence Asynchrone (Concurrency & Threading)
- **Le Problème** : Les *Vision Transformers* (ViT) pour l'estimation de l'âge nécessitent un pré-traitement asymétrique complexe (redimensionnement 224x224, normalisation ImageNet, transposition BGR vers RGB) et ont un temps d'inférence coûteux (`>100ms`). Exécutés dans la boucle principale (synchrone), ils bloquaient le flux vidéo (bottleneck E/S).
- **La Solution** : Implémentation du multithreading (`threading` module). Création d'une instance `AsyncViTEngine`. L'image (frame) courante est copiée dynamiquement (`copy.deepcopy()`) via un Mutex (`threading.Lock()`) et analysée par un *worker thread* en arrière-plan. Un mécanisme de signal (`threading.Event()`) notifie la UI lorsque les probabilités (Softmax) sont prêtes pour l'affichage, garantissant l'intégrité de l'UX.

### C. Ingénierie Inverse d'un Biais de Genre (Hybrid Inference Engine)
- **Le Problème** : Intégration d'un modèle SOTA (State of the Art) basé sur ViT, pré-entraîné sur le dataset massif *UTKFace* pour l'âge et le genre. Lors des tests empiriques, le modèle souffrait d'un grave effondrement de la classe `Male` (prédisant constamment `Female`). L'analyse des tenseurs de sortie (`logits`) a prouvé un biais positif intrinsèque aux poids du modèle, que la fonction d'activation (`sigmoid`) ne pouvait pas rattraper par un simple ajustement de seuil (Thresholding).
- **La Solution** : Refactoring d'urgence vers un **Moteur Hybride**. Le ViT a été conservé uniquement pour son excellence en régression (Estimation de l'Âge MAE ~4.5 ans). Le sous-problème de classification binaire du genre a été déporté vers un réseau de neurones convolutif dédié (`Caffe CNN`) qui a prouvé sa fiabilité statistique. L'ordonnateur exécute ces deux processus inférentiels de front (Side-by-side inference).

### D. Reconnaissance Locale Continue (Few-Shot Face Embeddings)
- **Le Mécanisme** : Utilisation d'un modèle basé sur un réseau résiduel profond capable d'obtenir des mesures encodant les visages dans l'espace euclidien vectoriel 128D (Face Embeddings).
- **Implémentation** : Implémentation d'une fonction d'apprentissage *Few-Shot* qui permet à l'utilisateur de fournir 1 image cible à la volée. Lors de l'inférence via `dlib` (`face_recognition`), la distance vectorielle est calculée entre l'encode courant et la matrice connue. Si `euclidean_distance < 0.60`, l'identification est un succès, prouvant qu'un système d'identification strict ne requiert pas un fine-tuning lourd pour chaque nouvel individu.

## 4. Modules Logiciels Annexes (La Super-Vision)

- **Coach Postural (MediaPipe Landmarks)** : Algorithme algorithmique extracteur de landmarks, analysant la géométrie squelettique (Angles Oeil ↔ Épaule). Calcul d'un seuil heuristique en `y` (profondeur) pour alerter la posture "dos voûté", mettant l'IA au service de l'ergonomie.
- **Le "Privacy Shield" (Traitement Aval / Downstream)** : Intégration native des notions de RGPD et d'éthique de l'IA. Si la phase de Face ID `euclidean_distance > 0.60` (Visage inconnu), l'image est passée à une fonction de filtrage pass-bas en aval (`cv2.GaussianBlur` 99x99 pixels). Ainsi, l'identité des tiers est cryptée mathématiquement avant d'être visualisée.

## 5. Conclusion 

AG Vision a évolué d'un simple script de détection Caffe en un micro-système complet, robuste, et optimisé en ressources. Ce projet met en valeur qu'au-delà du simple appel de bibliothèques ML ou de l'entraînement brut de poids, l'ingénierie moderne de l'IA réside dans le **déploiement système** (System Design), la synchronisation de modèles asymétriques (Threading, Multi-Models Pipeline), l'optimisation matérielle, et la capacité à isoler et contourner analytiquement les limites inhérentes des modèles State-of-the-Art (Gestion des Biais algorithmiques via Approches Hybrides).
