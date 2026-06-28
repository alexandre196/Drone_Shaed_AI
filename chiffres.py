from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/shahed_detector/weights/best.pt')
    metrics = model.val(workers=0)

    print("\n=== MÉTRIQUES PAR CLASSE ===")
    for i, name in model.names.items():
        p  = metrics.box.p[i]
        r  = metrics.box.r[i]
        f1 = 2*p*r/(p+r)
        ap = metrics.box.ap50[i]
        print(f"{name:10} | P={p:.4f} | R={r:.4f} | F1={f1:.4f} | mAP50={ap:.4f}")

    print("\n=== GLOBAL ===")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP@50-95 : {metrics.box.map:.4f}")