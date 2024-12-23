import argparse
import logging
import os
import threading
import queue
import yaml

import cv2
import pandas as pd

from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit

from model_inference import Embedding_generator
from image_processing import Image_processer
from search import HNSW_search_tool
from pgconnector import PGDBconnector
from logging_config import setup_logging 

### FLASK 
app = Flask(__name__)
socketio = SocketIO(app)

### LOGGING
setup_logging()
app_logger = logging.getLogger('app_logger')

### ARGPARSE
parser = argparse.ArgumentParser(description="MTG Card Detector")
parser.add_argument("-cip", "--camera_ip", type=str, default='192.168.0.101:8080', help="Camera IP with the port clarified")
parser.add_argument("-hnsw", "--hnsw_folder", type=Path, default=Path(r'data\embeddings'), help="A folder with HNSW bin file and metadata json")
parser.add_argument("-pg", "--postgres_config", type=Path, default=Path(r'data\pg_config.yaml'), help="A postgress config file")
args = parser.parse_args()
CAMERA_IP_ADDRESS = args.camera_ip
HNSW_FOLDER = args.hnsw_folder
with open(args.postgres_config, 'r') as file:
    CONNECTION_PARAMS = yaml.load(file, Loader=yaml.loader.SafeLoader)

### OTHER 
CAMERA_URL = f'http://{CAMERA_IP_ADDRESS}/video'
CARDS_RECOGNIZED = []
detection_frame_queue = queue.Queue(maxsize=1)
median_frame_queue = queue.Queue(maxsize=1)
processed_frame_queue = queue.Queue(maxsize=1)
results_queue = queue.Queue(maxsize=10)

def get_contours_if_stable():

    while True: 

        frame = detection_frame_queue.get()
        contour_image, contours = image_processer.find_big_contours(frame)
        
        if image_processer.similarity > 0.85 and contours:
            try:
                image_processer.camera_stable_flag = True
                freezing_frame = cv2.blur(frame, (25, 25), 0)
                freezing_frame = cv2.putText(freezing_frame, 'Processing', (frame.shape[1] // 4, frame.shape[0] // 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
                processed_frame_queue.put(freezing_frame)

                warped_images = [image_processer.crop_warp_image_from_contour(frame, contour) for contour in contours]
                embeddings = model_interface.generate_image_embedding(warped_images)
                results = [hnsw_search.search_in_hnsw(embedding, k=200)[0] for embedding in embeddings]
                [results_queue.put(result) for result in results]
                image_processer.camera_stable_flag = False
                image_processer.sliding_window.clear()
            except Exception:
                app_logger.error("An error occurred", exc_info=True)
                processed_frame_queue.put(contour_image)
        else:
            processed_frame_queue.put(contour_image)

def gen_frames(camera_url):
    
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
            socketio.emit('new_card', result)
        
        if not processed_frame_queue.empty():
            frame = processed_frame_queue.get()
    
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/delete_card', methods=['POST'])
def delete_card():
    card_number = request.json.get('card_number')

    global CARDS_RECOGNIZED
    CARDS_RECOGNIZED = [card for card in CARDS_RECOGNIZED if int(card["card_number"]) != card_number]
    socketio.emit('update_cards', CARDS_RECOGNIZED)

    return jsonify({"success": True, "card_number": card_number})

@app.route('/upload_table', methods=['POST'])
def upload_table():
    
    # Parse JSON data from the request
    data = request.get_json()
    if 'table' in data:
        table_data = data['table']
        print("Received table data:", table_data)

        return jsonify({"success": True, "received_data": table_data})
    else:
        return jsonify({"success": False, "message": "No table data received."}), 400
    
@app.route('/')
def index():
    # The home page with the video feed
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route that streams frames with both original and perspective-corrected images
    return Response(gen_frames(CAMERA_URL),  # Replace with your IP webcam URL
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        pgconnector = PGDBconnector(CONNECTION_PARAMS)
        model_interface = Embedding_generator()
        hnsw_search = HNSW_search_tool(768, 'cosine', HNSW_FOLDER / 'hnsw_index_cos.bin', 50, HNSW_FOLDER / 'image_emb_metadata.json')
        image_processer = Image_processer(median_frame_queue)
        app_logger.info("Set up succesfully")
        socketio.run(app, host='0.0.0.0', port=5000)
        app_logger.info("Connection Closed by User")
    except Exception:
        app_logger.error("An error occurred", exc_info=True)