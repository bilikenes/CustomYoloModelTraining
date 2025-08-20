from ultralytics import YOLO

def train_model():

    model = YOLO('yolov12n.pt')
    results = model.train(
        data='C:/Users/PC/Desktop/plates/dataset/data.yaml',
        epochs=50,
        device=0
    )

if __name__ == "__main__":
    train_model()