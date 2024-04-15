from ultralytics import YOLO

# model = YOLO('yolov8x.pt') # pass any model type
# results = model.train(data="/home/labs/training/class46/Aerial-IR-REID/yolov8/data.yaml", epochs=50, save_period=1)


model = YOLO("/home/labs/training/class46/Aerial-IR-REID/yolov8/runs/detect/train4/weights/last.pt")
results = model.train(resume=True)