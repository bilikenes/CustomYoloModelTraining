from ultralytics import YOLO

def train_model():

    model = YOLO('yolov12n.pt')
    results = model.train(
        data=r'D:\Medias\dataset_create\for_dataset\data.yaml',
        epochs=50,
        imgsz=640,              
        batch=16,              
        name="terminal_model_4" ,
    )

if __name__ == "__main__":
    train_model()