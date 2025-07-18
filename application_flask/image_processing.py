from collections import deque  # noqa: D100, INP001
from queue import Queue

import cv2
import numpy as np
from skimage.metrics import structural_similarity


class ImageProcesser:
    """Using for processing images in a thread."""

    def __init__(self, median_frame_queue: Queue) -> None:  # noqa: D107
        self.sliding_window = deque(maxlen=25)

        # Flags
        self.camera_stable_flag: bool = False
        self.similarity: float = 0.0
        self.median_frame_queue = median_frame_queue
        print("Image Processer has loaded")

    def is_camera_stable(self) -> None:
        """Works in a thread and detect whether the frame is stable."""
        while True:
            if self.camera_stable_flag:
                continue
            frame = self.median_frame_queue.get()
            frame_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (0, 0), fx=0.1, fy=0.1)
            frame_gray = cv2.GaussianBlur(frame_gray, (15, 15), 0)
            self.sliding_window.append(frame_gray)
            if len(self.sliding_window) < 25:
                self.similarity = 0.0
                continue
            accumulated = np.mean(self.sliding_window, axis=0).astype(np.uint8)
            self.similarity = structural_similarity(frame_gray, accumulated, full=False)

    def find_big_contours(self, frame: np.ndarray, epsilon_factor: float=0.05, min_area: int=50_000) -> tuple[np.ndarray, list]:
        """Detect contours on an image using OpenCV and keeps only those that are rectangular-like.

        Args:
            frame (np.array): The frame captured from the camera on which contours are detected.
            epsilon_factor (float, optional): Factor for approximating arcs in contours.
                A lower value results in a more accurate approximation. Defaults to 0.02.
            min_area (int, optional): Minimum area for contours to be kept. This is
                empirically determined to filter out smaller, irrelevant contours. Defaults to 2000.

        Returns:
            Tuple[np.array, list]: A tuple where:
                - The first element is the image with detected contours drawn on it.
                - The second element is a list of contours that match the specified criteria.

        """

        def _calculate_nonzero_ratio(thresh: np.ndarray) -> float:
            non_zero_pixels = cv2.countNonZero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            return non_zero_pixels / total_pixels

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        nonzero_ratio = _calculate_nonzero_ratio(thresh)
        if nonzero_ratio > 0.5:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_to_keep = []

        for contour in contours:
            epsilon = epsilon_factor * cv2.arcLength(contour, closed=True)
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            area = cv2.contourArea(contour)
            if len(approx) != 4 or area < min_area:
                continue
            contours_to_keep.append(contour)

        contour_image = frame.copy()
        cv2.drawContours(contour_image, contours_to_keep, -1, (0, 255, 0), 15)

        return contour_image, contours_to_keep

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in the correct order (tl, tr, br, bl)."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def crop_warp_image_from_contour(
        self,
        image: np.ndarray,
        contour: np.ndarray,
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Crops a specified contour from an image and applies perspective correction.

        Args:
            image (np.ndarray): _description_
            contour (np.ndarray): _description_
            target_size (tuple[int, int] | None, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_

        """
        epsilon = 0.05 * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        approx = self._order_points(approx.reshape(4, 2)).astype("float32")
        (tl, tr, br, bl) = approx

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        width = max(int(width_top), int(width_bottom))

        height_right = np.linalg.norm(br - tr)
        height_left = np.linalg.norm(bl - tl)
        height = max(int(height_right), int(height_left))

        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(approx, destination)
        warped_image = cv2.warpPerspective(image, M, (width, height))

        if target_size:
            warped_image = cv2.resize(warped_image, target_size)

        return warped_image
