import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import os
from datetime import datetime
import numpy as np

class DeepCRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(True)
        )
        self.rnn = nn.LSTM(256 * (imgH//8), nh, bidirectional=True, num_layers=2)
        self.embedding = nn.Linear(nh*2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.permute(3,0,2,1)
        conv = conv.reshape(w, b, -1)
        rnn_out, _ = self.rnn(conv)
        output = self.embedding(rnn_out)
        return output

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
        conv = conv.permute(3,0,2,1)
        conv = conv.reshape(w, b, -1)
        rnn_out, _ = self.rnn(conv)
        output = self.embedding(rnn_out)
        return output

characters = '0123456789ABCDEFGHIJKLMNOPRSTUVYZ'
idx_to_char = {idx+1: char for idx, char in enumerate(characters)}
idx_to_char[0] = ''

def decode(preds):
    prev = -1
    result = ''
    for p in preds:
        if p != prev and p != 0:
            result += idx_to_char[p.item()]
        prev = p
    return result

nclass = len(characters)+1

crnn_normal = CRNN(64, 1, nclass, 256)
crnn_normal.load_state_dict(torch.load(r"D:\OCR Model\EasyOCRTraining\turkish_plate_crnn.pth", map_location="cpu"))
crnn_normal.eval()

crnn_square = DeepCRNN(128, 1, nclass, 256)
crnn_square.load_state_dict(torch.load(r"D:\OCR Model\EasyOCRTraining\deep_square_plate_crnn.pth", map_location="cpu"))
crnn_square.eval()

transform_square = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_normal = transforms.Compose([
    transforms.Resize((64, 128)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

yolo_model = YOLO(r"D:\Yolo Model\CustomYoloModelTraining\runs\detect\train4\weights\best.pt")

def is_square_plate(pil_image):
    w, h = pil_image.size
    proportion = w / h
    return proportion <= 2

input_folder = r"D:\Medias\test_datas_for_YOLO"
output_folder = r"D:\Medias\results"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plate_text_final = "Okunamadı"

    if image is None:
        print(f"Could not read image: {image_path}. Saving as Okunamadı-{timestamp}.jpg")
        save_name = f"Okunamadı-{timestamp}.jpg"
        save_path = os.path.join(output_folder, save_name)
        try:
            from PIL import Image
            orig_img = Image.open(image_path)
            orig_img.save(save_path)
        except:
            print(f"Failed to save {image_path}, skipping.")
        continue

    results = yolo_model(image, conf=0.5)
    max_conf = -1
    best_plate_crop = None
    best_box = None
    ocr_model_name = ""

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        if len(boxes) == 0:
            continue
        for i, box in enumerate(boxes):
            if confidences[i] > max_conf:
                max_conf = confidences[i]
                best_plate_crop = image[box[1]:box[3], box[0]:box[2]]
                best_box = box

    if best_plate_crop is not None:
        pil_img = Image.fromarray(cv2.cvtColor(best_plate_crop, cv2.COLOR_BGR2GRAY))

        if is_square_plate(pil_img):
            input_img = transform_square(pil_img).unsqueeze(0)
            ocr_model = crnn_square
            ocr_model_name = "kare_plaka_modeli"
        else:
            input_img = transform_normal(pil_img).unsqueeze(0)
            ocr_model = crnn_normal
            ocr_model_name = "normal_plaka_modeli"

        with torch.no_grad():
            outputs = ocr_model(input_img)
            outputs = outputs.softmax(2)
            preds = outputs.argmax(2).squeeze(1)
            plate_text = decode(preds)
            if plate_text != '':
                plate_text_final = plate_text

        x1, y1, x2, y2 = best_box
        cv2.rectangle(image, (x1, y1), (x2, y2), (174, 52, 235), 2)
        cv2.putText(image, plate_text_final, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (174, 52, 235), 2, cv2.LINE_AA)
        cv2.putText(image, ocr_model_name, (image.shape[1]-10-250, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        print(f"No plate detected in {filename}, saving as Okunamadı-{timestamp}.jpg")

    save_name = f"{plate_text_final}-{timestamp}.jpg"
    save_path = os.path.join(output_folder, save_name)
    cv2.imwrite(save_path, image)
    print(f"Saved: {save_path}")

