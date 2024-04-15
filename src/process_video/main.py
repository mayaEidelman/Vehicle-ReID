import cv2
import torch

from services.embeddings.dvir import OnnxEmbeddingService
# from services.embeddings.reid_dino import DinoV2REIDEmbeddingService
# from services.embeddings.greid import GREIDEmbeddingService

from services.yolo import YoloService, BoundingBox, YoloObject

# from process_video.services.embeddings.dino import DinoService
from services.vector_similarity import VectorSimilarityService
from services.video import VideoService
import os

import argparse


def prepare_video_dir(video_path):
    base_name = os.path.basename(video_path)
    dir_name = os.path.splitext(base_name)[0]

    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def create_video(dir_name, vec_sim_result):
    pass


def index_video(yolo_service, dino_service, vector_similarity_service, video_path):
    object_counter = 1
    dir_name = prepare_video_dir(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_num = 1
    while cap.isOpened():
        success, frame = cap.read()
        print("frame_number =", frame_num)
        frame_num += 1
        if success:
            yolo_results = yolo_service.process_image(frame)
            cv2.imwrite(f"{dir_name}/{frame_num-1}.png", frame)
            with open(f"{dir_name}/{frame_num-1}.txt", "w") as file:
                for res in yolo_results:
                    res.counter = object_counter
                    res.frame = frame_num
                    file.write(repr(res) + "\n")
                    cropped_img = frame[
                        res.bbox.y1 : res.bbox.y2, res.bbox.x1 : res.bbox.x2
                    ]

                    embedding = dino_service.get_embedding(image=cropped_img)
                    vector_similarity_service.add_vector_to_index(embedding)
                    object_counter += 1
        else:
            break

    cap.release()
    print("saving index...")
    vector_similarity_service.save_index()


def search_image(
    query_image,
    dino_service,
    vector_similarity_service: VectorSimilarityService,
    video_service: VideoService,
):
    object_counter = 0

    embedding = dino_service.get_embedding(image_path=query_image)
    vec_sim_result = vector_similarity_service.search(embedding)
    print(vec_sim_result)
    vec_sim_result = vec_sim_result["indexes"][0]

    for filename in sorted(
        os.listdir(video_service.dir_name), key=lambda x: int(x.split(".")[0])
    ):
        if filename.endswith(".txt"):
            filepath = os.path.join(video_service.dir_name, filename)

            with open(filepath, "r") as file:
                contents = [line.strip() for line in file]

                for line in contents:
                    yolo_obg = eval(line)
                    image_path = filepath.replace(".txt", ".png")
                    image = cv2.imread(image_path)
                    
                    # Define rectangle parameters
                    x1, y1 = (
                        yolo_obg.bbox.x1,
                        yolo_obg.bbox.y1,
                    )  # Top-left corner coordinates
                    x2, y2 = (
                        yolo_obg.bbox.x2,
                        yolo_obg.bbox.y2,
                    )  # Bottom-right corner coordinates
                    color = (255, 0, 0)
                    thickness = 2  # Line thickness (pixels)
                    if object_counter in vec_sim_result:
                        color = (0, 0, 255)
                        print("#" * 60)
                        print(yolo_obg.track_id)
                        # Draw the rectangle on the image
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                    cv2.imwrite(image_path, image)
                    object_counter += 1

    print("creating video")
    video_service.create_video()

    # create_video(vec_sim_result)


if __name__ == "__main__":

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Description of your script.")

    # Add arguments
    parser.add_argument(
        "--search", action="store_true", help="Description of optional_arg"
    )
    parser.add_argument(
        "--dim", type=int, default=1024, help="Description of optional_arg"
    )
    parser.add_argument(
        "--index", type=str, default="vector.index", help="Description of optional_arg"
    )
    # Parse the arguments
    args = parser.parse_args()

    # Global Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model_path = "/home/labs/training/class46/Aerial-IR-REID/src/object_detector/yolov8/runs/detect/train4/weights/best.pt"

    video_path = "/home/labs/training/class46/Aerial-IR-REID/src/process_video/videos/good2.mp4"
    query_image_path = "query2.png"

    # Searching phase
    is_searching = args.search

    # Initialize Services
    yolo_service = YoloService(model_path=yolo_model_path)
    # dino_service = DinoService(device=device)
    model_path = "/home/labs/training/class46/Aerial-IR-REID/external_gits/fast-reid/onnx/baseline.onnx"
    dino_service = OnnxEmbeddingService(model_path)
    # dino_service = DinoV2REIDEmbeddingService()
    # dino_service = GREIDEmbeddingService()

    vector_similarity_service = VectorSimilarityService(load=is_searching, dim=args.dim, index=args.index)
    video_service = VideoService(video_path)

    if not is_searching:
        index_video(yolo_service, dino_service, vector_similarity_service, video_path)

    else:
        print("start searching")
        search_image(
            query_image_path, dino_service, vector_similarity_service, video_service
        )
