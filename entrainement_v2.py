# -*- coding: utf-8 -*-
"""
FINE-TUNING SHAHED DETECTOR v2
═══════════════════════════════
Améliore la détection des vues de dessus et de dessous du Shahed-136.
Repart du modèle best.pt existant (transfer learning).

Usage :
    python entrainement_v2.py

Résultat :
    runs/detect/shahed_v2/weights/best.pt  ← nouveau modèle amélioré
"""

import os
import shutil
from pathlib import Path

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
BASE_MODEL = "runs/detect/shahed_detector/weights/best.pt"
DATA_YAML  = "shahed_dataset/data.yaml"
EPOCHS     = 50
IMGSZ      = 640
BATCH      = 8
LR0        = 0.001
FREEZE     = 10
PATIENCE   = 15
PROJECT    = "runs/detect"
NAME       = "shahed_v2"
DEVICE     = ""


def main():
    from ultralytics import YOLO

    print("=" * 60)
    print("  FINE-TUNING SHAHED DETECTOR v2")
    print("=" * 60)

    # Vérifications
    if not os.path.exists(BASE_MODEL):
        print(f"\n[ERREUR] Modèle introuvable : {BASE_MODEL}")
        input("Appuie sur Entrée pour quitter..."); return
    print(f"[OK] Modèle de base     : {BASE_MODEL}")

    if not os.path.exists(DATA_YAML):
        print(f"\n[ERREUR] data.yaml introuvable : {DATA_YAML}")
        input("Appuie sur Entrée pour quitter..."); return
    print(f"[OK] Dataset            : {DATA_YAML}")

    n_train = len(list(Path("shahed_dataset/train/images").glob("*.*")))
    n_val   = len(list(Path("shahed_dataset/valid/images").glob("*.*")))
    print(f"[OK] Images train       : {n_train}")
    print(f"[OK] Images validation  : {n_val}")

    new_imgs = (
        len(list(Path("shahed_dataset/train/images").glob("vlcsnap*"))) +
        len(list(Path("shahed_dataset/train/images").glob("frame_*")))
    )
    print(f"[OK] Nouvelles images   : {new_imgs} (vues dessus/dessous)")

    output_dir = Path(PROJECT) / NAME
    if output_dir.exists():
        print(f"\n[WARN] Dossier existant : {output_dir}")
        rep = input("  Écraser ? (o/n) : ").strip().lower()
        if rep == "o":
            shutil.rmtree(output_dir)

    print()
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch      : {BATCH}")
    print(f"  LR0        : {LR0}")
    print(f"  Freeze     : {FREEZE} couches")
    print(f"  Patience   : {PATIENCE}")
    print()
    input("Tout est prêt — appuie sur Entrée pour lancer le fine-tuning...")

    print("\n[INFO] Chargement du modèle...")
    model = YOLO(BASE_MODEL)

    print("[INFO] Démarrage du fine-tuning...\n")
    model.train(
        data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        lr0=LR0, lrf=0.01, freeze=FREEZE, patience=PATIENCE,
        project=PROJECT, name=NAME, device=DEVICE, exist_ok=True,
        flipud=0.3, fliplr=0.5, degrees=15.0, translate=0.1,
        scale=0.3, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, mosaic=0.5,
        save=True, save_period=10, plots=True, verbose=True,
        workers=4,
    )

    # Validation finale
    best_path = Path(PROJECT) / NAME / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  RÉSULTATS DU FINE-TUNING")
    print("=" * 60)

    if not best_path.exists():
        print("[ERREUR] best.pt non trouvé")
        input("Entrée pour quitter..."); return

    print(f"\n[OK] Nouveau modèle : {best_path}")

    print("\n[INFO] Validation nouveau modèle...")
    m_new     = YOLO(str(best_path))
    met_new   = m_new.val(data=DATA_YAML, imgsz=IMGSZ, verbose=False)
    map50_new = met_new.box.map50

    print("\n[INFO] Validation ancien modèle...")
    m_old     = YOLO(BASE_MODEL)
    met_old   = m_old.val(data=DATA_YAML, imgsz=IMGSZ, verbose=False)
    map50_old = met_old.box.map50

    print(f"\n  Ancien mAP@50 : {map50_old*100:.1f}%")
    print(f"  Nouveau mAP@50: {map50_new*100:.1f}%")
    delta = (map50_new - map50_old) * 100
    if delta >= 0:
        print(f"  Amélioration  : +{delta:.1f}% ✅")
    else:
        print(f"  Régression    : {delta:.1f}% ⚠️")

    print("\n" + "=" * 60)
    if map50_new >= map50_old:
        print("  ✅ NOUVEAU MODÈLE MEILLEUR")
        rep = input("  Remplacer best.pt automatiquement ? (o/n) : ").strip().lower()
        if rep == "o":
            dest = Path("runs/detect/shahed_detector/weights")
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(BASE_MODEL, dest / "best_v1_backup.pt")
            shutil.copy(best_path, dest / "best.pt")
            print(f"[OK] Ancien sauvegardé : best_v1_backup.pt")
            print(f"[OK] Nouveau installé  : best.pt")
            print("\n  🎯 Relance le Shahed Detection System !")
    else:
        print("  ⚠️  Garde les deux et teste manuellement sur s1.mp4")
        print(f"  Nouveau : {best_path}")

    input("\nAppuie sur Entrée pour quitter...")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()