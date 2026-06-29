# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║        TÉLÉCHARGEMENT DATASET SHAHED — Roboflow API              ║
║  Auteur : alexandre196                                           ║
║                                                                  ║
║  INSTALLATION :                                                  ║
║    pip install roboflow                                          ║
║                                                                  ║
║  USAGE :                                                         ║
║    1. Remplace YOUR_API_KEY_ICI par ta clé privée Roboflow       ║
║    2. Lance : python download_shahed_dataset.py                  ║
║    3. Le dataset sera dans ./shahed_dataset/                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

from roboflow import Roboflow

# ⚠️  — ne jamais partager ce fichier
API_KEY = "YOUR_API_KEY_HERE"

print("Connexion à Roboflow...")
rf = Roboflow(api_key=API_KEY)

print("Téléchargement du dataset shahed (7 600 images)...")
project = rf.workspace("e-yjnj4").project("shahed-y4fsd")

# Version 4 = la dernière version disponible
dataset = project.version(4).download(
    model_format="yolov8",
    location="./shahed_dataset"
)

print(f"\n✅ Dataset téléchargé dans : {dataset.location}")
print(f"   Classes : {dataset.classes}")
print("\nStructure du dossier :")
print("  shahed_dataset/")
print("  ├── train/")
print("  │   ├── images/")
print("  │   └── labels/")
print("  ├── valid/")
print("  │   ├── images/")
print("  │   └── labels/")
print("  ├── test/")
print("  │   ├── images/")
print("  │   └── labels/")
print("  └── data.yaml")
print("\nProchain script : merge_datasets.py")