import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import os

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
idx_to_char[0] = '' 

def decode(preds):
    prev = -1
    result = ''
    for p in preds:
        if p != prev and p != 0:
            result += idx_to_char[p.item()]
        prev = p
    return result

nclass = len(characters) + 1
crnn_model = CRNN(32, 1, nclass, 256)
crnn_model.load_state_dict(torch.load('D:/Test Files/turkish_plate_crnn.pth', map_location='cpu'))
crnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

yolo_model = YOLO(r"D:\Python Files\yolov12\runs\detect\train2\weights\best.pt")

image_path = r"D:\Medias\plates\01\28\06AV4382-00-08-26.jpg"
image = cv2.imread(image_path)

results = yolo_model.predict(source=image, conf=0.5)

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # [x1,y1,x2,y2]
    
    for box in boxes:
        x1, y1, x2, y2 = box
        plate_crop = image[y1:y2, x1:x2]

        pil_img = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY))
        input_img = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            outputs = crnn_model(input_img)
            outputs = outputs.softmax(2)
            preds = outputs.argmax(2).squeeze(1)
            plate_text = decode(preds)

        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(image, plate_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2, cv2.LINE_AA)

        print(f"Plate : {plate_text}")

save_path = r"D:\Medias\plates\output\result_2.jpg"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
cv2.imwrite(save_path, image)
print("ok : ", save_path)
