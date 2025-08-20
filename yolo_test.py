import cv2
import os
from ultralytics import YOLO

model = YOLO(r"C:\Users\PC\Desktop\Yolo Model\CustomYoloModelTraining\runs\detect\train4\weights\best.pt")

save_dir = r"C:\Users\PC\Desktop\plates\detected_plates\01\31"
os.makedirs(save_dir, exist_ok=True)

results = model.predict(
    source=r"C:\Users\PC\Desktop\plates\01\31",
    conf=0.50,
    save=False
)

count = 0
for r in results:
    img = cv2.imread(r.path)

    best_box = None
    best_conf = 0.0

    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 0 and conf > best_conf:
            best_conf = conf
            best_box = box

    base_name = os.path.basename(r.path)

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Plate {best_conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        base_name = "1_" + base_name

    filename = os.path.join(save_dir, base_name)
    cv2.imwrite(filename, img)
    count += 1

print(f"ok : {count}")