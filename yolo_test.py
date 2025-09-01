import cv2
import os
import random
from ultralytics import YOLO

model = YOLO(r"runs\detect\terminal_model_1\weights\best.pt")

save_dir = r"D:\Medias\night_images"
os.makedirs(save_dir, exist_ok=True)

source_dir = r"D:\Medias\fotograflar_Karasu_Belediyesi_Foto\night_photos"

files = os.listdir(source_dir)
random.shuffle(files)

count = 0
for file in files:
    print(file)
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(source_dir, file)
    img_path = os.path.join(source_dir, file)

    if not os.path.exists(img_path):
        print(f"[ERROR] File not found: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read image: {img_path}")
        continue

    results = model.predict(source=img_path, conf=0.50, save=False, verbose=False)


    best_box = None
    best_conf = 0.0

    r = results[0]
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 0 and conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Plate {best_conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        filename = os.path.join(save_dir, file)
    else:
        filename = os.path.join(save_dir, f"okunamadi_{file}")

    cv2.imwrite(filename, img)
    print(f"{count}-) ok : {filename}")
    count += 1

print(f"total : {count}")