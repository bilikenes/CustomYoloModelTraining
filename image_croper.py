# import cv2
# import os
# from ultralytics import YOLO

# model = YOLO(r"runs\detect\train4\weights\best.pt")

# save_dir = r"D:\Medias\plates\detected_plates\01\29"
# os.makedirs(save_dir, exist_ok=True)

# results = model.predict(
#     source=r"D:\Medias\plates\01\29",
#     conf=0.50,
#     save=False
# )

# count = 0
# for r in results:
#     best_box = None
#     best_conf = -1

#     for box in r.boxes:
#         cls = int(box.cls[0])
#         conf = float(box.conf[0])

#         if cls == 0 and conf > best_conf:
#             best_conf = conf
#             best_box = box

#     if best_box is not None:
#         x1, y1, x2, y2 = map(int, best_box.xyxy[0])
#         img = cv2.imread(r.path)
#         plate_crop = img[y1:y2, x1:x2]

#         base_name = os.path.basename(r.path)
#         plate_name = base_name.split("-")[0] + ".jpg"

#         filename = os.path.join(save_dir, plate_name)
#         cv2.imwrite(filename, plate_crop)
#         count += 1

# print(f"total : {count} and OK")

for i in range(1,10):
    print("31")

import os
import cv2
from ultralytics import YOLO
model = YOLO(r"D:\Yeni klasör\CustomYoloModelTraining\runs\detect\train4\weights\best.pt")
root_dir = r"D:\Medias\plates\07"
save_root = r"D:\Medias\plates\detected_plates\07"
total_count = 0
for subdir, dirs, files in os.walk(root_dir):
    if not files:
        continue
    relative_path = os.path.relpath(subdir, root_dir)
    save_dir = os.path.join(save_root, relative_path)
    os.makedirs(save_dir, exist_ok=True)
    results = model.predict(
        source=subdir,
        conf=0.50,
        save=False
    )
    count = 0
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
            base_name = os.path.basename(r.path)
            plate_name = base_name.split("-")[0] + ".jpg"
            filename = os.path.join(save_dir, plate_name)
            cv2.imwrite(filename, plate_crop)
            count += 1
    total_count += count
print(f"ok : {total_count}")

