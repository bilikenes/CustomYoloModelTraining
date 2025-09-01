import cv2
import numpy as np
import os
from ultralytics import YOLO

import cv2
import numpy as np

def straighten_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        if w < h:
            angle = angle + 90

        if abs(angle) > 30:
            (h_img, w_img) = plate_crop.shape[:2]
            center = (w_img // 2, h_img // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            plate_crop = cv2.warpAffine(plate_crop, M, (w_img, h_img),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)

            W, H = 300, 100
            dst_pts = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(np.float32(box), dst_pts)
            warped = cv2.warpPerspective(plate_crop, M, (W, H))
            return warped

    return plate_crop


model = YOLO(r"runs\detect\terminal_model_1\weights\best.pt")
root_dir = r"D:\Medias\fotograflar_Karasu_Belediyesi_Foto\day_photos\02"
save_root = r"D:\Medias\test_datas\01\04"
total_count = 0

for subdir, dirs, files in os.walk(root_dir):
    if not files:
        continue
    relative_path = os.path.relpath(subdir, root_dir)
    save_dir = os.path.join(save_root, relative_path)
    os.makedirs(save_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(subdir, file)
        results = model.predict(source=file_path, conf=0.50, save=False)

        for r in results:
            best_box = None
            best_conf = 0.0
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > 0.50 and conf > best_conf:
                    best_conf = conf
                    best_box = box

            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                img = cv2.imread(r.path)
                plate_crop = img[y1:y2, x1:x2]

                plate_warped = straighten_plate(plate_crop)

                base_name = os.path.basename(r.path)
                plate_name = base_name.split("-")[0] + ".jpg"
                filename = os.path.join(save_dir, plate_name)
                cv2.imwrite(filename, plate_warped)
                print(f"saved warped plate: {filename}")
                total_count += 1

print(f"Toplam kaydedilen plaka: {total_count}")
