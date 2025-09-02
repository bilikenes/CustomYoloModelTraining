from ultralytics import YOLO

def train_model():

    model = YOLO('yolov12n.pt')

    model.train(
        data=r'C:\Users\PC\Desktop\plates\dataset\data.yaml',
        epochs=200,
        imgsz=640,        
        batch=16,
        lr0=0.01,
        lrf=0.2,
        optimizer='SGD',
        augment=True,
        device=0,
        project='runs/exp',
        name='plates_yolov12',
    )

if __name__ == "__main__":
    train_model()