import torch
from ultralytics import YOLO

def train_yolo():

    # Load the model with Group Normalization
    model = YOLO("yolov8.yaml").load("../yolov8n.pt")

    # Print the normalization type, default to 'batch' if not specified
    print(f'\nNormalization type: {model.model.yaml.get("norm_type", "batch")} normalization\n')

    # Run the model training
    results = model.train(data="coco128.yaml", epochs=1, imgsz=640, batch=4)

    # Load the trained model
    model = YOLO("best.pt")

    # Load an image
    image = torch.rand(1, 3, 640, 640)

    # Run the model inference
    result = model(image)


if __name__ == '__main__':
    train_yolo()
