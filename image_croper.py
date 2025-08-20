import cv2
import os
from ultralytics import YOLO

# model = YOLO('yolov12n.pt')

# results = model.train(
#     data=f'{"D:/Medias/dataset"}/data.yaml',
#     epochs=10
# )

# model = YOLO(r"C:\Users\PC\Desktop\Yolo Model\CustomYoloModelTraining\runs\detect\train4\weights\best.pt")

# save_dir = r"C:\Users\PC\Desktop\plates\detected_plates\01\21"
# os.makedirs(save_dir, exist_ok=True)

# results = model.predict(
#     source=r"C:\Users\PC\Desktop\plates\01\21",
#     conf=0.50,
#     save=False
# )

# count = 0
# for r in results:
#     for box in r.boxes:
#         cls = int(box.cls[0])
#         conf = float(box.conf[0])

#         if cls == 0 and conf > 0.50:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             img = cv2.imread(r.path)
#             plate_crop = img[y1:y2, x1:x2]

#             base_name = os.path.basename(r.path) 
#             plate_name = base_name.split("-")[0] + ".jpg"  

#             filename = os.path.join(save_dir, plate_name)
#             cv2.imwrite(filename, plate_crop)
#             count += 1

# print(f"total :  {count} and OK")


import os
import cv2
from ultralytics import YOLO

# Model yükle
model = YOLO(r"C:\Users\PC\Desktop\Yolo Model\CustomYoloModelTraining\runs\detect\train4\weights\best.pt")

# Ana klasör (ör: plates dizini)
root_dir = r"C:\Users\PC\Desktop\plates\07"
save_root = r"C:\Users\PC\Desktop\plates\detected_plates\07"

total_count = 0

# Tüm klasörleri dolaş
for subdir, dirs, files in os.walk(root_dir):
    if not files:  # içi boşsa geç
        continue

    # Kaydedilecek klasör yolunu oluştur (root_dir'den farkını alıp save_root ile birleştiriyoruz)
    relative_path = os.path.relpath(subdir, root_dir)
    save_dir = os.path.join(save_root, relative_path)
    os.makedirs(save_dir, exist_ok=True)

    # Model tahmini yap
    results = model.predict(
        source=subdir,
        conf=0.50,
        save=False
    )

    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.50:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                img = cv2.imread(r.path)
                plate_crop = img[y1:y2, x1:x2]

                base_name = os.path.basename(r.path)
                plate_name = base_name.split("-")[0] + ".jpg"

                filename = os.path.join(save_dir, plate_name)
                cv2.imwrite(filename, plate_crop)
                count += 1

    print(f"{relative_path} klasöründe {count} plaka bulundu ve kaydedildi.")
    total_count += count

print(f"Toplam tespit edilen plaka sayısı: {total_count}")
