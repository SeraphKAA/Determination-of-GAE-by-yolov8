from ultralytics import YOLO


model = YOLO("yolov8n.pt")

# из-за того что после 50-60 эпохи не идет толком улучшение, поэтому с 100 эпох перехожу на 60
results = model.train(data="my_dataset_ethnos_yolov8/data.yaml", epochs=60, imgsz=640, device="cpu")
# results = model.train(data="my_dataset_yolov8/data.yaml", epochs=100, imgsz=640)