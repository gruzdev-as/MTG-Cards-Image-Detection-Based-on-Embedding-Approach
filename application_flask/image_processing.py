import cv2 
import numpy as np
import time 

from collections import deque
from skimage.metrics import structural_similarity
from typing import Tuple

class Image_processer:

    def __init__(self, median_frame_queue):

        self.sliding_window = deque(maxlen=25)
        
        # Flags 
        self.camera_stable_flag = False
        self.similarity = 0
        self.median_frame_queue = median_frame_queue
        print('Image Processer has loaded')

    def is_camera_stable(self):
        """
        Works in a thread and detect whether the frame is stable 
        """

        while True: 
            if not self.camera_stable_flag: 
                frame = self.median_frame_queue.get()
                frame_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (0, 0), fx=0.1, fy=0.1)
                self.sliding_window.append(frame_gray)
                if len(self.sliding_window) < 25:
                    self.similarity = 0
                    continue
                accumulated = np.mean(self.sliding_window, axis=0).astype(np.uint8)
                self.similarity, _ = structural_similarity(frame_gray, accumulated, full=True)
                           
    
    def find_big_contours(self, 
        frame: np.array, 
        epsilon_factor: float=0.02, 
        min_area: int=2000, 
        aspect_ratio_range: Tuple[float, float]=(0.5, 2.0)) -> Tuple[np.array, list]:
        
        """
        Detects contours on an image using OpenCV and keeps only those that are rectangular-like.

        Args:
            frame (np.array): The frame captured from the camera on which contours are detected.
            epsilon_factor (float, optional): Factor for approximating arcs in contours.
                A lower value results in a more accurate approximation. Defaults to 0.02.
            min_area (int, optional): Minimum area for contours to be kept. This is 
                empirically determined to filter out smaller, irrelevant contours. Defaults to 2000.
            aspect_ratio_range (Tuple[float, float], optional): Range of acceptable aspect ratios 
                (width-to-height ratio) for contours to be kept. Defaults to (0.5, 2.0), favoring 
                contours that are roughly rectangular.

        Returns:
            Tuple[np.array, list]: A tuple where:
                - The first element is the image with detected contours drawn on it.
                - The second element is a list of contours that match the specified criteria.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 5)
        thresh = cv2.dilate(thresh, None, iterations=5) / 255
        thresh = cv2.convertScaleAbs(thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_contours = []

        for contour in contours:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)

            if len(approx) != 4 or area < min_area: 
                continue

            _, _, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                rectangular_contours.append(contour)

        contour_image = frame.copy()
        cv2.drawContours(contour_image, rectangular_contours, -1, (0, 255, 0), 15)

        return contour_image, rectangular_contours
           
    def crop_warp_image_from_contour(self, image:np.array, contour:list) -> np.array:
        """
        Crops a specified contour from an image and applies perspective correction to minimize distortions.

        Args:
            image (np.array): The original image frame from which the contour will be cropped.
            contour (list): The contour to be cropped and warped, represented as a list of points.

        Returns:
            np.array: The cropped and perspective-corrected (warped) section of the image.
        """

        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        width = max(np.linalg.norm(approx[2] - approx[3]), np.linalg.norm(approx[1] - approx[0]))
        height = max(np.linalg.norm(approx[1] - approx[2]), np.linalg.norm(approx[0] - approx[3]))

        destination_points = np.array([
            [0, 0],  
            [width - 1, 0],  
            [width - 1, height - 1],  
            [0, height - 1]  
        ], dtype='float32')

        M = cv2.getPerspectiveTransform(np.float32(approx), destination_points)
        warped_image = cv2.warpPerspective(image, M, (int(width), int(height)))
        warped_image = cv2.flip(warped_image, 1)
        if warped_image.shape[0] < warped_image.shape[1]:
            warped_image = cv2.rotate(warped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
        warped_image = cv2.resize(warped_image, (480, 680))

        return warped_image