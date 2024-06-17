from ultralytics import YOLO  # Assuming you are using ultralytics YOLO implementation

def train_yolo():
    model = YOLO("yolov8.yaml").load("yolov8n.pt")  # build from YAML and transfer weights
    results = model.train(data="coco128.yaml", epochs=100, imgsz=640, batch=4)
    return results

if __name__ == '__main__':
    train_yolo()
