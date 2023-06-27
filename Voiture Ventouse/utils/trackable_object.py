""" Trackable Object Class """

import time
from collections import deque

import torch
from utils.custom_type import Point, Rectangle

class TrackableObject:
    """ Trackable Object Class """
    def __init__(self, object_id: int, centroid: Point, rect: Rectangle, tensor: torch.Tensor):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.object_id = object_id
        self.centroids = deque(maxlen=20)
        self.centroids.append(centroid)
        self.rect = rect
        self.tensor = tensor
        self.apparition = time.time()
        self.last_apparition = self.apparition
        self.reported = False

    def update(self, centroid, rect, tensor, distance):
        """Update object.

        Args:
            centroid (tuple): The centroid of object.
            rect (tuple): The rect of object.
            tensor (numpy.ndarray): The tensor of object.
        """
        self.centroids.append(centroid)
        self.tensor = tensor
        if distance > 30:
            self.rect = rect
        self.last_apparition = time.time()
