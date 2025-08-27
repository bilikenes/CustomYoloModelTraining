from ultralytics import YOLO

def train_model():

    model = YOLO("yolov12n.pt")

    result = model.train(
        data=r"C:\Users\PC\Desktop\dataset_for_YOLO\data.yaml", 
        epochs=150,             
        imgsz=640,              
        batch=16,              
        name="terminal_model_3" ,
    )

if __name__ == "__main__":
    train_model()