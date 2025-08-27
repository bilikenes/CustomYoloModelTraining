import cv2
import os
from ultralytics import YOLO

model = YOLO(r"runs\detect\terminal_model_1\weights\best.pt")

save_dir = r"C:\Users\PC\Desktop\detected_plates"
os.makedirs(save_dir, exist_ok=True)

results = model.predict(
    source=r"C:\Users\PC\Desktop\fotograflar_Karasu_Belediyesi_Foto",
    conf=0.50,
    save=False
)

count = 0
for r in results:
    best_box = None
    best_conf = -1

    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 0 and conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        img = cv2.imread(r.path)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        base_name = os.path.basename(r.path)
        plate_name = base_name.split("-")[0] + ".jpg"

        filename = os.path.join(save_dir, plate_name)
        cv2.imwrite(filename, img)
        count += 1

print(f"total : {count} and OK")
