""" Centroid Tracker Class """
import io
import logging
import time
from collections import OrderedDict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import distance as dist # type: ignore

from .custom_type import Config, Point, Rect
from .func import calculate_iou, image_similarity, send_image_s3, send_xml_s3
from .trackable_object import TrackableObject


class CentroidTracker:
    """Centroid Tracker Class"""

    def __init__(self, camera: str, config: tuple, time_dispappeared=10, similarity=0.8):
        """__init__.

        Args:
            camera (str): The camera name.
            config (tuple): The AWS config tuple.
            time_dispappeared (int): The time in seconds before an object is considered disappeared. Defaults to 10
            similarity (float): The similarity threshold. Defaults to 0.8.
        """
        self.camera = camera
        self.next_object_id = 0
        self.objects: OrderedDict[int, TrackableObject] = OrderedDict()
        self.similarity = similarity
        self.time_dispappeared = time_dispappeared
        self.config = Config(access_key=config[0], secret_key=config[1], bucket_name=config[2])
        self.webhook_teams = config[3]

    def register(self, centroid: Point, rect: Rect, tensor: np.ndarray, image: np.ndarray) -> None:
        """Register new object to tracking list.

        Args:
            centroid (Point): The centroid of object.
            rect (Rect): The bounding box of object.
            tensor (ndarray): The tensor of object.
            image (ndarray): The image of object.
        """
        logging.info("%s | Register: %d", self.camera, self.next_object_id)
        self.objects[self.next_object_id] = TrackableObject(
            self.next_object_id, centroid, rect, tensor, image
        )
        self.next_object_id += 1

    def deregister(self, object_id: int, reason="disappeared"):
        """Delete object from tracking list.

        Args:
            object_id (int): The object id to be deleted.
            reason (str, optional): The reason of deletion. Defaults to "disappeared".
        """
        logging.info("%s | Deregister: %d for reason: %s", self.camera, object_id, reason)
        if reason == "disappeared" and not self.objects[object_id].sended:
            self.send(object_id)
            
        del self.objects[object_id]

    def send(self, object_id: int):
        """Send object to S3.

        Args:
            object_id (int): The object id to be sended.
        """
        logging.info("%s | Send: %d", self.camera, object_id)
        truck = self.objects[object_id]
        image = Image.fromarray(cv2.cvtColor(truck.image, cv2.COLOR_BGR2RGB))
        image_data = io.BytesIO()
        image.save(image_data, format="JPEG")
        # convert the timestamp to a datetime object in the local timezone
        dt_object = datetime.fromtimestamp(truck.apparition)
        date = dt_object.strftime("%Y-%m-%d_%H-%M-%S")
        send_xml_s3(truck.original_rect, truck.image.shape , f"truck/3.5T/Annotations/{self.camera}-{date}.xml", self.config)
        send_image_s3(image_data.getvalue(), f"truck/3.5T/JPEGImages/{self.camera}-{date}.jpg", self.camera, self.config, self.webhook_teams)
        truck.sended = True


    def delete_double(self, iou_threshold: int):
        """Delete double objects.

        Args:
            iou_threshold (int): The threshold of IoU.
        """
        if len(self.objects) < 2:
            return
        # Initialize list to store filtered objects
        list_objects_id = list(self.objects.keys())
        filtered_objects = set(list_objects_id)

        # Loop through objects
        for i, detection in enumerate(self.objects.values()):
            # Compare overlap with remaining objects
            for j in list_objects_id[i + 1 :]:
                overlap = calculate_iou(detection.original_rect, self.objects[j].original_rect)
                #print("Overlap: ", overlap, "for objects: ", detection.object_id, "and ", self.objects[j].object_id)
                # Remove overlapping rectangle if IoU exceeds threshold
                if overlap > iou_threshold:
                    #print("Remove: ", self.objects[j].object_id)
                    filtered_objects.remove(self.objects[j].object_id)
                    break

        #print("Should keep: ", filtered_objects)

        remove = set(self.objects.keys()) - filtered_objects
        #print("Remove: ", remove)
        for object_id in remove:
            self.deregister(object_id, "double")

    def rects_to_centroid(self, rects: list) -> np.ndarray:
        """Convert rects to centroid.

        Args:
            rects (list): The list of rects.

        Returns:
            np.ndarray: The list of centroids.
        """
        centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (start_x, start_y, end_x, end_y) in enumerate(rects):
            mid_x = int((start_x + end_x) / 2.0)
            mid_y = int((start_y + end_y) / 2.0)
            centroids[i] = Point(mid_x, mid_y)
        return centroids

    def update(self, tracks: list, frame: np.ndarray):
        """Update tracking list.

        Args:
            tracks (list): The list of tracks.
            frame (ndarray): The frame.

        Returns:
            list: The list of objects
        """
        if len(tracks) == 0:
            for object_id, truck in self.objects.copy().items():
                truck.frame_disappeared += 1
                if time.time() - truck.last_apparition > self.time_dispappeared:
                    self.deregister(object_id)
            return self.objects

        rects = list(map(lambda x: x["bbox"], tracks))
        tensors = list(map(lambda x: x["tensor"], tracks))
        input_centroids = self.rects_to_centroid(rects)

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, rects[i], tensors[i], frame)

        else:
            list_object_id = list(self.objects.keys())
            list_object_centroids = list(
                map(lambda x: x.centroids[-1], self.objects.values())
            )

            distance = dist.cdist(np.array(list_object_centroids), input_centroids)

            rows = distance.min(axis=1).argsort()
            cols = distance.argmin(axis=1)[rows]
            used_tensors = set()
            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                for i, tensor in enumerate(tensors):
                    if tensor in used_tensors:
                        continue
                    object_id = list_object_id[row]
                    cos_sim_value = image_similarity(
                        tensor, self.objects[object_id].tensor
                    )
                    if cos_sim_value > self.similarity:
                        self.objects[object_id].update(
                            input_centroids[col], rects[col], tensor, frame # type: ignore
                        )
                        used_rows.add(row)
                        used_cols.add(col)
                        used_tensors.add(tensor)
                        if time.time() - self.objects[object_id].apparition > 10 and not self.objects[object_id].sended:
                            self.send(object_id)
                        break
                # object_id = list_object_id[row]
                # # Calculate the distance between the previous centroid and the updated centroid
                # previous_centroid = list_object_centroids[row]
                # updated_centroid = input_centroids[col]
                # centroid_distance = np.sqrt(np.sum((updated_centroid - previous_centroid)**2))

                # self.objects[object_id].update(input_centroids[col], rects[col], cars[0], centroid_distance)
                # used_rows.add(row)
                # used_cols.add(col)

            unused_rows = set(range(0, distance.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distance.shape[1])).difference(used_cols)

            if distance.shape[0] >= distance.shape[1]:
                for row in unused_rows:
                    object_id = list_object_id[row]
                    self.objects[object_id].frame_disappeared += 1
                    if (time.time() - self.objects[object_id].last_apparition) > self.time_dispappeared:
                        self.deregister(object_id, "disappeared")
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], rects[col], tensors[col], frame)

        return self.objects
