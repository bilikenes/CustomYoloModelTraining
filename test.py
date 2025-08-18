import cv2
import os
from ultralytics import YOLO

# model = YOLO('yolov12n.pt')

# results = model.train(
#     data=f'{"D:/Medias/dataset"}/data.yaml',
#     epochs=10
# )

model = YOLO(r"D:\Python Files\yolov12\runs\detect\train2\weights\best.pt")

save_dir = r"D:\Medias\plates\detected_plates\detected_plates_03_04"
os.makedirs(save_dir, exist_ok=True)

results = model.predict(
    source=r"D:\Medias\plates\03\04",
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

print(f"total :  {count} and OK")
