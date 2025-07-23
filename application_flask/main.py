import argparse
import logging
import queue
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal

import cv2
import yaml
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO
from image_processing import ImageProcesser
from logging_config import setup_logging
from model_inference import EmbeddingGenerator
from pgconnector import PGDBconnector
from search import HNSWSearchTool

### FLASK
app = Flask(__name__)
socketio = SocketIO(app)

### LOGGING
setup_logging()
app_logger = logging.getLogger("app_logger")

### ARGPARSE
parser = argparse.ArgumentParser(description="MTG Card Detector")
parser.add_argument("-cip", "--camera_ip", type=str, default="192.168.0.102:8080", help="Camera IP with the port clarified")
parser.add_argument("-hnsw", "--hnsw_folder", type=Path, default=Path(r"data\embeddings"), help="A folder with HNSW bin file and metadata json")
parser.add_argument("-pg", "--postgres_config", type=Path, default=Path(r"data\pg_config.yaml"), help="A postgress config file")
args = parser.parse_args()
CAMERA_IP_ADDRESS = args.camera_ip
HNSW_FOLDER = args.hnsw_folder
with Path(args.postgres_config).open("r") as file:
    CONNECTION_PARAMS = yaml.load(file, Loader=yaml.loader.SafeLoader)

### OTHER
CAMERA_URL = f"http://{CAMERA_IP_ADDRESS}/video"
CARDS_RECOGNIZED = []
detection_frame_queue = queue.Queue(maxsize=1)
median_frame_queue = queue.Queue(maxsize=1)
processed_frame_queue = queue.Queue(maxsize=1)
results_queue = queue.Queue(maxsize=10)


def get_contours_if_stable() -> None:
    while True:
        frame = detection_frame_queue.get()
        contour_image, contours = image_processer.find_big_contours(frame)
        if image_processer.similarity > 0.75 and contours:
            try:
                image_processer.camera_stable_flag = True
                freezing_frame = cv2.GaussianBlur(frame, (25, 25), 0)
                freezing_frame = cv2.putText(
                    img=freezing_frame,
                    text="Processing",
                    org=(frame.shape[1] // 4, frame.shape[0] // 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                processed_frame_queue.put(freezing_frame)

                warped_images = [image_processer.crop_warp_image_from_contour(frame, contour, (480, 680)) for contour in contours]
                embeddings = model_interface.generate_image_embedding(warped_images)
                results = [hnsw_search.search_in_hnsw(embedding, k=200)[0] for embedding in embeddings]
                [results_queue.put(result) for result in results]
                image_processer.camera_stable_flag = False
                image_processer.sliding_window.clear()
            except Exception:
                app_logger.exception("An error occurred")
                processed_frame_queue.put(contour_image)
        else:
            processed_frame_queue.put(contour_image)


def gen_frames(camera_url:str) -> Generator[Any]:
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print("Error: Unable to open video stream")
        return

    threading.Thread(target=image_processer.is_camera_stable, daemon=True).start()
    threading.Thread(target=get_contours_if_stable, daemon=True).start()

    while True:
        success, frame = cap.read()

        if not success:
            break

        if not image_processer.camera_stable_flag:
            detection_frame_queue.put(frame)
            median_frame_queue.put(frame)

        if not results_queue.empty():
            result = results_queue.get()
            CARDS_RECOGNIZED.append(result)
            socketio.emit("new_card", result)

        if not processed_frame_queue.empty():
            frame = processed_frame_queue.get()

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/delete_card", methods=["POST"])
def delete_card() -> Response:
    idx_to_remove = request.json.get("row_index")

    CARDS_RECOGNIZED.pop(idx_to_remove)
    socketio.emit("update_cards", CARDS_RECOGNIZED)

    return jsonify({"success": True, "card_number": idx_to_remove})


@app.route("/upload_table", methods=["POST"])
def upload_table() -> Response | tuple[Response, Literal[400]]:
    # Parse JSON data from the request
    data = request.get_json()
    table_data = data["table"]
    try:
        for card_item_data, card_data in zip(table_data, CARDS_RECOGNIZED,  strict=True):
            processed_card_data = pgconnector.prepare_card_data(card_data)
            card_id = pgconnector.add_card(processed_card_data)
            card_item_data["card_id"] = card_id
            pgconnector.update_or_create_inventory(**card_item_data)
        return jsonify({"success": True, "received_data": table_data})
    except Exception:
        app_logger.exception("An error while uploading")
        return jsonify({"success": False, "message": "No table data received."}), 400


@app.route("/")
def index() -> str:
    # The home page with the video feed
    return render_template("index.html")


@app.route("/video_feed")
def video_feed() -> Response:
    # Video streaming route that streams frames with both original and perspective-corrected images
    return Response(
        gen_frames(CAMERA_URL),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    try:
        pgconnector = PGDBconnector(CONNECTION_PARAMS)
        model_interface = EmbeddingGenerator()
        hnsw_search = HNSWSearchTool(768, "cosine", HNSW_FOLDER / "hnsw_index_cos.bin", 50, HNSW_FOLDER / "image_emb_metadata_new.json")
        image_processer = ImageProcesser(median_frame_queue)
        app_logger.info("Set up succesfully")
        socketio.run(app, host="0.0.0.0", port=5000)
        pgconnector.close_connection()
        app_logger.info("Connection Closed by User")
    except Exception:
        app_logger.exception("An error occurred")
