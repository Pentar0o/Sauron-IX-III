""" Trackable Object Class """

import colorsys
import time
from collections import deque, namedtuple

Vector = namedtuple('Vector', ['x', 'y', 'direction_x', 'direction_y'])

class TrackableObject:
    """ Trackable Object Class """
    def __init__(self, object_id, centroid, original_rect, tensor):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.object_id = object_id
        self.centroids = deque(maxlen=20)
        self.centroids.append(centroid)
        self.original_rect = original_rect
        self.tensor = tensor
        self.apparition = time.time()
        self.last_apparition = self.apparition

    def color(self):
        """Get color from first apperance of object.
        Args:
           

        Returns:
            tuple: The color generated from string.
        """
        return value_to_rgb(time.time() - self.apparition)

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
            self.original_rect = rect
        self.last_apparition = time.time()

def normalize_seconds(value):
    """Normalize a value to the range [0, 1000].

    Args:
        value (float): Value to normalize.

    Returns:
        float: Normalized value.
    """
    min_value = 0
    max_value = 7 * 60 * 60 # 7 days in seconds
    min_normalized = 0
    max_normalized = 1000

    normalized_value = ((value - min_value) / (max_value - min_value)) * (max_normalized - min_normalized) + min_normalized
    return normalized_value

def value_to_rgb(seconds):
    """Convert a value to a RGB color.

    Args:
        data (seconds): Value to convert.

    Returns:
        tuple: RGB color.
    """
    # Normalize data to the range [0, 1000]
    data = normalize_seconds(seconds)  

    # Calculate the hue and value
    hue = data / 1000
    value = 1 - data / 1000

    # Convert the HSV color to RGB
    red, green, blue = colorsys.hsv_to_rgb(hue, 1, value)
    red = int(red * 255)
    green = int(green * 255)
    blue = int(blue * 255)
    return red, green, blue