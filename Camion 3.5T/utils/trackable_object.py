""" Trackable Object Class """

import time
from collections import deque, namedtuple
import numpy as np

from .func import calculate_bbox_area
from .custom_type import Point, Rect

class TrackableObject:
    """ Trackable Object Class """
    def __init__(self, object_id: int, centroid: Point, original_rect: Rect, tensor: np.ndarray, image: np.ndarray):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.object_id = object_id
        self.centroids = deque(maxlen=20)
        self.centroids.append(centroid)
        self.original_rect = original_rect
        self.tensor = tensor
        self.image = image
        self.apparition = time.time()
        self.last_apparition = self.apparition
        self.sended = False
        self.frame_disappeared = 0

    def update(self, centroid: Point, rect: Rect, tensor: np.ndarray, image: np.ndarray):
        """Update object.

        Args:
            centroid (tuple): The centroid of object.
            rect (tuple): The rect of object.
            tensor (numpy.ndarray): The tensor of object.
            image (numpy.ndarray): The image of object.
        """
        self.centroids.append(centroid)
        self.tensor = tensor
        self.last_apparition = time.time()
        self.frame_disappeared = 0
        areaA = calculate_bbox_area(self.original_rect)
        areaB = calculate_bbox_area(rect)
        if areaB > areaA :
            self.image = image
            self.original_rect = rect
