import sys
from ultralytics import YOLO
m = YOLO("yolo11n.pt")
m.train(data=sys.argv[1], imgsz=320, epochs=45, patience=12, batch=32, device="mps", workers=4,
        project=sys.argv[2], name="chars", exist_ok=True, cos_lr=True, close_mosaic=8,
        fliplr=0.0, degrees=3, shear=2, perspective=0.0005, hsv_v=0.5, hsv_s=0.5, scale=0.3, translate=0.08,
        mosaic=0.5, plots=True, verbose=False)
