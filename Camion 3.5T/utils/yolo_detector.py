""" YoloDetector class
"""

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .func import calculate_bbox_area, image_to_tensor


class YoloDetector:
    def __init__(
        self,
        model_path,
        device,
        confidence,
        division,
        area_threshold,
        classes
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.division = division
        self.area_threshold = area_threshold
        if self.model.names is not None:
            model_classes = dict(filter(
                lambda x: x[1] in classes, self.model.names.items()
            ))
            self.classes = list(model_classes.keys())
        else:
            self.classes = []


    def detect(self, frame: np.ndarray, cam_index: int) -> list:
        """Detect truck in a frame using YOLOv8.

        Args:
            model (YOLO): YOLO model.
            frame (ndarray): Frame from the video stream.
            cam_index (int): Camera index.

        Returns:
            tracks (list): List of tracks.
        """

        # Number of parts to divide the frame into
        num_lines = 10

        tracks = []
        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform car detection using Ultralytics
        results = self.model(rgb_frame, device=self.device, verbose=False)

        for result in results:
            # Loop through all detections found by YOLOv8
            for box in result.boxes:
                if (int(box.cls) in self.classes) and box.conf > self.confidence:
                    start_x, start_y, end_x, end_y = box.xyxy[0]
                    bbox_area = calculate_bbox_area(
                        [start_x, start_y, end_x, end_y]
                    )
                    if self.division:
                        min_y = min(start_y, end_y)
                        frame_height = frame.shape[0]
                        position = min_y // (frame_height // num_lines)
                    else:
                        position = 0
                    if bbox_area > (self.area_threshold * (1 + 0.1 * position)):
                        # crop the car
                        start_x = max(int(start_x), 0)
                        start_y = max(int(start_y), 0)
                        end_x = min(int(end_x), frame.shape[1])
                        end_y = min(int(end_y), frame.shape[0])
                        truck = frame[start_y:end_y, start_x:end_x]
                        # convert the car to tensor
                        tensor = image_to_tensor(
                            Image.fromarray(truck), self.device
                        )

                        # add the tracker to our list of trackers so we can
                        # utilize it during skip frames
                        tracks.append(
                            {
                                "bbox": (
                                    int(start_x),
                                    int(start_y),
                                    int(end_x),
                                    int(end_y),
                                ),
                                "cam_index": cam_index,
                                "tensor": tensor,
                            }
                        )
        return tracks
