# 🎯 Shahed Detection System
### YOLOv8 real-time drone detection — Kalman tracking · PDF report · KML export

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![License](https://img.shields.io/badge/License-GPL--3.0-red?style=flat-square)
![mAP](https://img.shields.io/badge/mAP%4050-91.1%25-brightgreen?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-lightgrey?style=flat-square)

> Système de détection temps réel de drones Shahed-136 par IA — développé par **alexandre196**

---

## 📸 Features

- ✅ **YOLOv8s** entraîné sur dataset Shahed — mAP@50 = **91.1%** (classes : `bird` / `not` / `shahed`)
- ✅ **Filtre de Kalman** + tracking multi-drones avec ID persistants
- ✅ **Live preview** dans l'interface (Tkinter)
- ✅ **Alarme sonore** temps réel (Windows) + intégrée dans la vidéo (ffmpeg)
- ✅ **Géolocalisation estimée** sans GPS — azimut + distance métrique
- ✅ **Mini-carte radar** live dans le GUI
- ✅ **Export KML** → Google Earth
- ✅ **Export CSV** → Excel / QGIS
- ✅ **Rapport PDF** avec stats, graphiques, carte et photo du drone
- ✅ **Analyse comportementale** : hovering, circling, approche rapide, erratique
- ✅ **Alertes email** (Gmail) + **push notifications** (Ntfy)
- ✅ **Calibration caméra** par objet de référence (en mètres)
- ✅ Sources : fichier vidéo / webcam / flux RTSP

---

## 🚀 Installation

```bash
pip install ultralytics opencv-python numpy matplotlib reportlab Pillow requests
```

**Optionnel — audio dans la vidéo :**
```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg
```

---

## 📥 Télécharger le modèle entraîné

Le fichier `best.pt` (mAP@50 = 91.1%) est disponible dans les **Releases** :

👉 **[Télécharger best.pt — v1.0](https://github.com/alexandre196/Drone_Shaed_AI/releases/download/v1.0/best.pt)**

Placer le fichier dans :
```
Drone_Shaed_AI/
└── runs/
    └── detect/
        └── shahed_detector/
            └── weights/
                └── best.pt   ← ici
```

---

## ▶️ Utilisation

```bash
python drone_shahed_detector.py
```

L'interface graphique s'ouvre automatiquement.

1. Sélectionner la **source vidéo** (fichier / webcam / RTSP)
2. Le modèle Shahed est chargé automatiquement
3. Cliquer **LANCER LA DÉTECTION**
4. Les fichiers sont générés dans un dossier horodaté

---

## 📁 Structure du projet

```
Drone_Shaed_AI/
├── drone_shahed_detector.py      # Application principale
├── train_shahed.py               # Script d'entraînement YOLOv8
├── download_shahed_dataset.py    # Téléchargement du dataset
├── Annotate_dessous.py           # Outil d'annotation
├── entrainement_v2.py            # Entraînement v2
├── chiffres.py                   # Statistiques dataset
├── target.png                    # Overlay viseur
├── .gitignore
└── runs/detect/shahed_detector/weights/best.pt  # Modèle (via Release)
```

---

## 📊 Performances du modèle

| Classe   | Précision | Rappel | mAP@50 |
|----------|-----------|--------|--------|
| bird     | 94.2%     | 91.8%  | 93.1%  |
| not      | 88.7%     | 86.4%  | 87.9%  |
| shahed   | 92.6%     | 93.4%  | 93.3%  |
| **All**  | **91.8%** | **90.5%** | **91.1%** |

---

## 🗂️ Fichiers générés après analyse

| Fichier | Description |
|---------|-------------|
| `*_detected.mp4` | Vidéo annotée avec HUD |
| `*_report.pdf` | Rapport complet avec stats et carte |
| `*_charts.png` | Graphiques distances / timeline |
| `*_geomap.png` | Carte des trajectoires estimées |
| `*_trajectory.kml` | Export Google Earth |
| `*_trajectory.csv` | Export Excel / QGIS |

---

## ⚙️ Paramètres principaux

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| Seuil danger | 300 m | Distance d'alerte en mètres |
| Frame skip | 1 | Traiter 1 frame sur N (performance) |
| Confirmation | 3 frames | Anti-faux-positifs |
| FOV caméra | 60° | Champ de vision horizontal |
| Cap caméra | 0° (Nord) | Orientation de la caméra |

---

## 📐 Distances par classe

| Classe | Largeur réelle | Usage |
|--------|---------------|-------|
| shahed / shahed-136 | 2.5 m | Envergure Shahed-136 |
| fpv-drone / drone | 0.40 m | FPV racing drone |
| bird / not | 0.50 m | Référence neutre |

---

## 📧 Alertes

- **Email** : Gmail avec App Password — alerte avec photo à chaque détection Shahed
- **Push** : [Ntfy.sh](https://ntfy.sh) — notification mobile instantanée

---

## 🛠️ Ré-entraîner le modèle

```bash
python train_shahed.py
```

Ou télécharger le dataset :
```bash
python download_shahed_dataset.py
```

---

## 📄 Licence

GPL-3.0 License — © 2026 Alexandre Martin (alexandre196)

---

*Développé à Lyon, France 🇫🇷*
