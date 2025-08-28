from ultralytics import YOLO

def train_model():

    model = YOLO("runs/detect/terminal_model_1/weights/best.pt")

    result = model.train(
        data=r"D:\Medias\dataset_for_train_YOLO_2\data.yaml", 
        epochs=50,             
        imgsz=640,              
        batch=16,              
        name="terminal_model_1_1" ,
    )

if __name__ == "__main__":
    train_model()