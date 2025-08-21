from ultralytics import YOLO
import cv2
import os
import csv
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128 * (imgH//4), nh, bidirectional=True, num_layers=2)
        self.embedding = nn.Linear(nh*2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.permute(3, 0, 2, 1)
        conv = conv.reshape(w, b, -1)
        rnn_out, _ = self.rnn(conv)
        output = self.embedding(rnn_out)
        return output

characters = '0123456789ABCDEFGHIJKLMNOPRSTUVYZ'
idx_to_char = {idx+1: char for idx, char in enumerate(characters)}
idx_to_char[0] = ''  # CTC blank

nclass = len(characters) + 1
ocr_model = CRNN(32, 1, nclass, 256)
ocr_model.load_state_dict(torch.load(r'C:\Users\PC\Desktop\OCR Model\EasyOCRTraining\turkish_plate_crnn.pth', map_location='cpu'))
ocr_model.eval()

ocr_transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def decode(preds):
    prev = -1
    result = ''
    for p in preds:
        if p != prev and p != 0:
            result += idx_to_char[p.item()]
        prev = p
    return result

def recognize_plate(img_bgr):
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    tensor_img = ocr_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        outputs = ocr_model(tensor_img)
        outputs = outputs.softmax(2)
        preds = outputs.argmax(2).squeeze(1)
        text = decode(preds)
    return text.strip()

model = YOLO(r"runs\detect\train4\weights\best.pt")
video_path = r"C:\Users\PC\Desktop\plates\0821_1.mp4"

save_dir = "detected_plates"
os.makedirs(save_dir, exist_ok=True)

csv_file = os.path.join(save_dir, "detections.csv")
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "plate_text", "class_id", "confidence", "time_sec", "x1", "y1", "x2", "y2"])

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
save_count = 0
frame_skip = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    current_time_sec = frame_count / fps
    print(f" {current_time_sec:.2f} saniye")

    frame_resized = cv2.resize(frame, (1280, 720)) 
    results = model.predict(frame_resized, conf=0.50, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.25:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame_resized[y1:y2, x1:x2]

                plate_text = recognize_plate(plate_img)
                if len(plate_text) > 2:  
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{save_dir}/{plate_text}-{timestamp}.jpg"
                else:
                    filename = f"{save_dir}/unknown-{save_count}.jpg"

                cv2.imwrite(filename, plate_img)

                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, plate_text, cls, conf, current_time_sec, x1, y1, x2, y2])

                print(f"[INFO] {current_time_sec:.2f}s → {plate_text} kaydedildi: {filename}")
                save_count += 1

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, plate_text,
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

    cv2.imshow("Detected Plates", frame_resized)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
