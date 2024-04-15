from ultralytics import YOLO
import os

# model = YOLO('yolov8x-obb.pt')
model = YOLO("/home/labs/training/class46/Aerial-IR-REID/src/object_detector/yolov8/runs/detect/train4/weights/best.pt")

results = model('/home/labs/training/class46/Aerial-IR-REID/src/object_detector/yolov8/datasets/zenodo/images/test/train112.jpg')  # return a list of Results objects
# path = "/home/labs/training/class46/Aerial-IR-REID/src/object_detector/yolov8/istockphoto-1420684404-640_adpp_is.mp4"
# results = model(path, stream=True)
# os.makedirs("./test1", exist_ok=True)
# metrics = model.val()
# Process results list
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    # print(probs)
    # breakpoint()
    # result.save(filename=f'./test1/{i}.jpeg', ) 