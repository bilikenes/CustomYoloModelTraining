import os
from ultralytics import YOLO
from PIL import Image

model = YOLO(r"runs\detect\terminal_model_1\weights\best.pt")

image_dir = r"C:\Users\PC\Desktop\detected_plates"
label_dir = r"C:\Users\PC\Desktop\detected_plates\labels"
os.makedirs(label_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    results = model.predict(source=img_path, conf=0.25, verbose=False)

    with Image.open(img_path) as im:
        w, h = im.size

    label_lines = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)  
            x1, y1, x2, y2 = box.xyxy[0]

            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(label_dir, txt_name)

    with open(txt_path, "w") as f:
        f.write("\n".join(label_lines))

    print(f"{txt_name} ok.")