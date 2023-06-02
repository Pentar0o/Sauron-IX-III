""" Centroid Tracker Class """
import logging
from collections import OrderedDict
import time
from scipy.spatial import distance as dist
from torchvision import transforms
import torch
import numpy as np

from .trackable_object import TrackableObject


class CentroidTracker:
    """Centroid Tracker Class"""

    def __init__(self, camera, time_dispappeared=10, similarity=0.8):
        """__init__.

        Args:
            camera (str): The camera name.
            time_dispappeared (int): The time in seconds before an object is considered disappeared. Defaults to 10
            similarity (float): The similarity threshold. Defaults to 0.8.
        """
        self.camera = camera
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.similarity = similarity
        self.time_dispappeared = time_dispappeared

    def register(self, centroid, rect, tensor):
        """Register new object to tracking list.

        Args:
            centroid (_type_): _description_
            rect (_type_): _description_
        """
        logging.info("%s | Register: %d", self.camera, self.next_object_id)
        self.objects[self.next_object_id] = TrackableObject(
            self.next_object_id, centroid, rect, tensor
        )
        self.next_object_id += 1

    def deregister(self, object_id, reason="disappeared"):
        """Delete object from tracking list.

        Args:
            object_id (int): The object id to be deleted.
        """
        logging.info("%s | Deregister: %d for reason: %s", self.camera, object_id, reason)
        del self.objects[object_id]


    def delete_double(self, iou_threshold):
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

    def rects_to_centroid(self, rects):
        """Convert rects to centroid.

        Args:
            rects (list): The list of rects.
        """
        centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (start_x, start_y, end_x, end_y) in enumerate(rects):
            mid_x = int((start_x + end_x) / 2.0)
            mid_y = int((start_y + end_y) / 2.0)
            centroids[i] = (mid_x, mid_y)
        return centroids

    def update(self, rects, cars):
        """Update tracking list.

        Args:
            rects (list): The list of rect of objects.
            cars (list): The list of cars tensor from frame.

        Returns:
            list: The list of objects
        """
        if len(rects) == 0:
            for object_id, car in self.objects.items():
                car.disappeared += 1
                car.direction = None
                if time.time() - car.last_apparition > self.time_dispappeared:
                    self.deregister(object_id)
            return self.objects


        input_centroids = self.rects_to_centroid(rects)

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, rects[i], cars[i])

        else:
            list_object_id = list(self.objects.keys())
            list_object_centroids = list(
                map(lambda x: x.centroids[-1], self.objects.values())
            )

            distance = dist.cdist(np.array(list_object_centroids), input_centroids)

            rows = distance.min(axis=1).argsort()
            cols = distance.argmin(axis=1)[rows]
            used_cars = set()
            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                for i, car in enumerate(cars):
                    if car in used_cars:
                        continue
                    object_id = list_object_id[row]
                    cos_sim_value = image_similarity(
                        car, self.objects[object_id].tensor
                    )
                    if cos_sim_value > self.similarity:
                        previous_centroid = list_object_centroids[row]
                        updated_centroid = input_centroids[col]
                        centroid_distance = np.sqrt(np.sum((updated_centroid - previous_centroid)**2))
                        self.objects[object_id].update(
                            input_centroids[col], rects[col], car, centroid_distance
                        )
                        used_rows.add(row)
                        used_cols.add(col)
                        used_cars.add(car)
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
                    if (time.time() - self.objects[object_id].last_apparition) > self.time_dispappeared:
                        self.deregister(object_id, "disappeared")
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], rects[col], cars[col])

            

        return self.objects


def image_similarity(image1_tensor, image2_tensor):
    """Calculate cosine similarity between two images.

    Args:
        image1_tensor (torch.Tensor): The first image tensor.
        image2_tensor (torch.Tensor): The second image tensor.

    Returns:
        float: The cosine similarity value.
    """
    # Flatten tensors
    image1_tensor = image1_tensor.flatten()
    image2_tensor = image2_tensor.flatten()
    # Calculate cosine similarity
    cos_sim_value = torch.nn.functional.cosine_similarity(
        image1_tensor, image2_tensor, dim=0
    ).item()
    return cos_sim_value

def calculate_iou(bbox1, bbox2):
    """Calculate intersection over union (IoU) between two bounding boxes.

    Args:
        bbox1 (tuple): The first bounding box.
        bbox2 (tuple): The second bounding box.

    Returns:
        float: The IoU value.
    """

    # Extract coordinates from bounding boxes
    start_x1, start_y1, end_x1, end_y1 = bbox1
    start_x2, start_y2, end_x2, end_y2 = bbox2

    # Calculate intersection coordinates
    intersection_x1 = np.maximum(start_x1, start_x2)
    intersection_y1 = np.maximum(start_y1, start_y2)
    intersection_x2 = np.minimum(end_x1, end_x2)
    intersection_y2 = np.minimum(end_y1, end_y2)

    # Calculate intersection area
    intersection_area = np.maximum(0, intersection_x2 - intersection_x1 + 1) * np.maximum(0, intersection_y2 - intersection_y1 + 1)

    # Calculate areas of bounding boxes
    bbox1_area = (end_x1 - start_x1 + 1) * (end_y1 - start_y1 + 1)
    bbox2_area = (end_x2 - start_x2 + 1) * (end_y2 - start_y2 + 1)

    # Calculate IoU
    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou
