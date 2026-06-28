# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║        REPRISE ENTRAÎNEMENT YOLOV8 — SHAHED THREAT DETECTOR      ║
║  Auteur : alexandre196                                           ║
║                                                                  ║
║  USAGE :                                                         ║
║    python train_shahed_resume.py                                 ║
║                                                                  ║
║  RÉSULTATS :                                                     ║
║    runs/detect/shahed_detector/weights/best.pt                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

from ultralytics import YOLO
import torch
import os

if __name__ == '__main__':

    # ── Vérification GPU ─────────────────────────────────────────
    print(f"CUDA disponible : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU  : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Checkpoint à reprendre ───────────────────────────────────
    LAST_PT = r"C:\Users\alexa\Desktop\Drone_Shaed_AI\runs\detect\shahed_detector\weights\last.pt"

    if not os.path.exists(LAST_PT):
        print(f"\n❌ Checkpoint non trouvé : {LAST_PT}")
        print("Vérifie le chemin vers last.pt")
        exit(1)
    else:
        print(f"\n✅ Checkpoint trouvé — reprise de l'entraînement !")

    # ── Reprise ──────────────────────────────────────────────────
    model = YOLO(LAST_PT)

    print(f"\n🚀 Reprise entraînement sur RTX 4070 Ti...\n")

    results = model.train(
        resume = True,   # <-- reprend exactement là où c'était arrêté
    )

    # ── Résultats ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ !")
    print("="*60)
    print(f"\n📁 Modèle sauvegardé :")
    print(f"   → runs/detect/shahed_detector/weights/best.pt")
    try:
        print(f"\n📊 Métriques finales :")
        print(f"   mAP@50    : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
        print(f"   Precision : {results.results_dict.get('metrics/precision(B)', 'N/A'):.3f}")
        print(f"   Recall    : {results.results_dict.get('metrics/recall(B)', 'N/A'):.3f}")
    except Exception:
        print("   (consulte runs/detect/shahed_detector/results.csv)")
    print(f"\n🎯 Copie best.pt dans ton dossier principal")
    print(f"   et pointe ton drone_detector.py vers ce nouveau modèle !")