# -*- coding: utf-8 -*-
"""
OUTIL D'ANNOTATION RAPIDE — vue de dessous Shahed
═══════════════════════════════════════════════════
Usage :
  1. Extrait les frames de s1.mp4 :
       ffmpeg -i s1.mp4 -vf fps=5 frames_dessous/frame_%04d.jpg

  2. Lance ce script :
       python annotate_dessous.py

  3. Pour chaque image :
       - Clique-glisse pour dessiner le rectangle autour du Shahed
       - S = Sauvegarder et image suivante
       - N = Passer sans annoter (image sans Shahed visible)
       - Z = Annuler la dernière bbox
       - Q = Quitter

Les fichiers .txt YOLO sont créés dans le même dossier que les images.
Classe shahed = 2 (bird=0, not=1, shahed=2)
"""

import cv2
import os
import glob
import sys

# ── CONFIG ────────────────────────────────────────────────────
IMAGES_DIR   = "SHAHED_dessous2"   # dossier avec les frames extraites
LABELS_DIR   = "SHAHED_dessous2"   # même dossier pour les .txt
SHAHED_CLASS = 2                  # index de shahed dans data.yaml
WIN_NAME     = "ANNOTATION — clic+glisse=bbox | S=save | N=skip | Z=annuler | Q=quitter"
MAX_WIDTH    = 1280               # largeur max fenêtre
MAX_HEIGHT   = 720

# ── Variables globales ────────────────────────────────────────
drawing   = False
ix, iy    = -1, -1
bboxes    = []      # liste de (x1,y1,x2,y2) en pixels
img_disp  = None    # image affichée (redimensionnée)
img_orig  = None    # image originale
scale     = 1.0     # facteur de redimensionnage


def draw_all(img):
    """Dessine toutes les bboxes enregistrées sur l'image."""
    out = img.copy()
    for (x1,y1,x2,y2) in bboxes:
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(out, "shahed", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return out


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, img_disp, bboxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        tmp = draw_all(img_disp)
        cv2.rectangle(tmp, (ix,iy), (x,y), (0,200,255), 1)
        cv2.imshow(WIN_NAME, tmp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix,x), min(iy,y)
        x2, y2 = max(ix,x), max(iy,y)
        if (x2-x1) > 5 and (y2-y1) > 5:
            bboxes.append((x1,y1,x2,y2))
        cv2.imshow(WIN_NAME, draw_all(img_disp))


def save_labels(img_path, img_w, img_h):
    """Sauvegarde les bboxes au format YOLO dans le .txt correspondant."""
    base    = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = LABELS_DIR
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, base + ".txt")

    lines = []
    for (x1,y1,x2,y2) in bboxes:
        # Reconvertir en coordonnées image originale
        rx1 = x1 / scale; ry1 = y1 / scale
        rx2 = x2 / scale; ry2 = y2 / scale
        # Format YOLO : classe cx cy w h (normalisé 0-1)
        cx = ((rx1+rx2)/2) / img_w
        cy = ((ry1+ry2)/2) / img_h
        bw = (rx2-rx1) / img_w
        bh = (ry2-ry1) / img_h
        # Clamp 0-1
        cx = max(0, min(1, cx)); cy = max(0, min(1, cy))
        bw = max(0, min(1, bw)); bh = max(0, min(1, bh))
        lines.append(f"{SHAHED_CLASS} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    return txt_path, len(lines)


def main():
    global img_disp, img_orig, scale, bboxes

    # Trouver les images
    exts = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]
    images = []
    for ext in exts:
        images += glob.glob(os.path.join(IMAGES_DIR, ext))
    images = sorted(set(images))

    if not images:
        print(f"[ERREUR] Aucune image trouvée dans '{IMAGES_DIR}'")
        print(f"  Extrait d'abord les frames :")
        print(f"  ffmpeg -i s1.mp4 -vf fps=5 {IMAGES_DIR}/frame_%04d.jpg")
        input("Appuie sur Entrée pour quitter...")
        return

    # Filtrer les images déjà annotées
    already = 0
    remaining = []
    for p in images:
        base = os.path.splitext(os.path.basename(p))[0]
        txt  = os.path.join(LABELS_DIR, base + ".txt")
        if os.path.exists(txt):
            already += 1
        else:
            remaining.append(p)

    print(f"[INFO] {len(images)} images trouvées")
    print(f"[INFO] {already} déjà annotées, {len(remaining)} restantes")

    if not remaining:
        print("[INFO] Toutes les images sont déjà annotées !")
        input("Appuie sur Entrée pour quitter...")
        return

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_NAME, mouse_cb)

    saved = skipped = 0
    i = 0

    while i < len(remaining):
        path = remaining[i]
        bboxes = []

        img_orig = cv2.imread(path)
        if img_orig is None:
            print(f"[WARN] Impossible de lire {path}")
            i += 1
            continue

        h, w = img_orig.shape[:2]

        # Redimensionner si trop grand
        scale = min(MAX_WIDTH/w, MAX_HEIGHT/h, 1.0)
        nw, nh = int(w*scale), int(h*scale)
        img_disp = cv2.resize(img_orig, (nw, nh)) if scale < 1.0 else img_orig.copy()

        # Barre de statut en haut
        status = img_disp.copy()
        cv2.rectangle(status, (0,0), (nw,28), (0,0,0), -1)
        txt = f"[{i+1}/{len(remaining)}] {os.path.basename(path)}  |  S=save  N=skip  Z=annuler  Q=quitter"
        cv2.putText(status, txt, (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,65), 1)
        img_disp = status

        cv2.imshow(WIN_NAME, draw_all(img_disp))

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s') or key == ord('S'):
                # Sauvegarder
                txt_path, n = save_labels(path, w, h)
                saved += 1
                print(f"[OK] {os.path.basename(path)} → {n} bbox(es) → {os.path.basename(txt_path)}")
                i += 1
                break

            elif key == ord('n') or key == ord('N'):
                # Skip sans annoter — crée un .txt vide (image négative)
                base    = os.path.splitext(os.path.basename(path))[0]
                txt_path = os.path.join(LABELS_DIR, base + ".txt")
                open(txt_path, "w").close()
                skipped += 1
                print(f"[SKIP] {os.path.basename(path)} → image négative")
                i += 1
                break

            elif key == ord('z') or key == ord('Z'):
                # Annuler la dernière bbox
                if bboxes:
                    bboxes.pop()
                    cv2.imshow(WIN_NAME, draw_all(img_disp))
                    print(f"[UNDO] {len(bboxes)} bbox(es) restante(s)")

            elif key == ord('q') or key == ord('Q') or key == 27:
                print(f"\n[INFO] Arrêt — {saved} sauvées, {skipped} skippées")
                cv2.destroyAllWindows()
                return

            elif cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                # Fenêtre fermée
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"\n[TERMINÉ] {saved} images annotées, {skipped} images négatives")
    print(f"Les .txt sont dans : {os.path.abspath(LABELS_DIR)}")
    print(f"\nProchaine étape : copie les images + .txt dans :")
    print(f"  shahed_dataset/train/images/  (images)")
    print(f"  shahed_dataset/train/labels/  (fichiers .txt)")
    input("Appuie sur Entrée pour quitter...")


if __name__ == "__main__":
    # Créer le dossier si besoin
    os.makedirs(IMAGES_DIR, exist_ok=True)
    main()