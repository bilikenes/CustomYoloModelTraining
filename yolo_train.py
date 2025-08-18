from ultralytics import YOLO

model = YOLO('yolov12n.pt')

results = model.train(
    data=f'{"D:/Medias/dataset"}/data.yaml',
    epochs=50
)