import sys
from ultralytics import YOLO
m = YOLO("yolo11s-cls.pt")
m.train(data=sys.argv[1], imgsz=224, epochs=14, patience=0, batch=64, device="mps", workers=4, save_period=1,
        project=sys.argv[2], name="helmet_mix", exist_ok=True, cos_lr=True, erasing=0.1, fliplr=0.5,
        hsv_v=0.4, scale=0.3, dropout=0.1, plots=False, verbose=False)
