import cv2
import os
import random
import numpy as np
from ultralytics import YOLO

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)

    kernel = np.ones((3,3), np.uint8)
    morphed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    return morphed

model = YOLO(r"runs\detect\terminal_model_1_2\weights\best.pt")

save_dir = r"D:\Medias\night_images"
os.makedirs(save_dir, exist_ok=True)

source_dir = r"D:\Medias\night_images\01"

files = os.listdir(source_dir)
random.shuffle(files)

count = 0
for file in files:
    print(file)
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(source_dir, file)

    if not os.path.exists(img_path):
        print(f"[ERROR] File not found: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read image: {img_path}")
        continue

    processed_img = preprocess_plate(img)
    processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    results = model.predict(source=processed_img_bgr, conf=0.50, save=False, verbose=False)

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
        cv2.rectangle(processed_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Plate {best_conf:.2f}"
        cv2.putText(processed_img_bgr, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        filename = os.path.join(save_dir, file)
    else:
        filename = os.path.join(save_dir, f"okunamadi_{file}")

    cv2.imwrite(filename, processed_img_bgr)
    print(f"{count}-) ok : {filename}")
    count += 1

print(f"total : {count}")
