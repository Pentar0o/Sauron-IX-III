import math
import cv2
import numpy as np
from collections import OrderedDict
from PIL import Image
import logging

def box_label(image: np.ndarray, box: tuple[int,int,int,int], label="", color: tuple[int,int,int] = (128, 128, 128)):
    """Draw a bounding box with a label on an image.

    Args:
        image (ndarray): Image to draw on.
        box (tuple): Bounding box coordinates.
        label (str, optional): Label to draw. Defaults to ''.
        color (tuple, optional): Color of the bounding box. Defaults to (128, 128, 128).
    """
    line_width = max(round(sum(image.shape) / 3 * 0.003), 2)
    left_top, right_bottom = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(
        image, left_top, right_bottom, color, line_width, cv2.LINE_AA # type: ignore 
    )
    if label:
        thickness = max(line_width - 1, 1)  # font thickness
        width, height = cv2.getTextSize(label, 0, line_width / 3, thickness)[
            0
        ]  # text width, height
        outside = left_top[1] - height >= 3
        if outside:
            right_bottom = left_top[0] + width, left_top[1] - height - 3
        else:
            right_bottom = left_top[0] + width, left_top[1] + height + 3
        cv2.rectangle( # filled
            image, left_top, right_bottom, color, -1, cv2.LINE_AA # type: ignore 
            )
        
        brightness_value = (299*color[0] + 587*color[1] + 114*color[2]) / 1000
        if brightness_value < 128:
            txt_color = (255, 255, 255)  # white text
        else:
            txt_color =  (0, 0, 0)  # black text
        cv2.putText(
            image,
            label,
            (left_top[0], left_top[1] - 2 if outside else left_top[1] + height + 2),
            0,
            line_width / 3,
            txt_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

def draw_tracks(frame: np.ndarray, trucks: OrderedDict):
    """Draw bounding boxes and directions of tracked trucks.

    Args:
        frame (ndarray): Frame from the video stream.
        trucks (OrderedDict): List of tracked trucks.
    """
    # loop over the tracked objects
    for object_id, car in trucks.items():
        if car.frame_disappeared == 0:
            rect = car.original_rect
            box_label(frame, rect, label=f"Truck {object_id}")

def make_image(images, camera, save):
    """Combine images from all cameras into a single image.

    Args:
        images (list): List of images from the cameras.
        camera (Camera): Camera object.
        save (bool): Whether to save the image.
    
    Returns:
        canvas (ndarray): Combined image.
    """
    # Resize the images to have the same height.
    max_height = max(image.size[1] for image in images)
    resized_images = []
    for image in images:
        height_ratio = max_height / image.size[1]
        new_width = int(image.size[0] * height_ratio)
        resized_image = image.resize((new_width, max_height), Image.LANCZOS)
        resized_images.append(resized_image)

    # Combine the resized images from all cameras into a single image.
    width = resized_images[0].size[1]
    height = resized_images[0].size[0]
    canvas = np.zeros((width * 2, height * 2, 3), dtype=np.uint8)
    canvas[:width, :height] = resized_images[0]
    canvas[:width, height:] = resized_images[1]
    canvas[width:, :height] = resized_images[2]
    canvas[width:, height:] = resized_images[3]
    
    if save:
        grid_image = Image.fromarray(canvas)
        camera_name = camera.name + "_Quad" if camera.quad else ""
        filename = f'{camera_name}-truck.jpg'
        grid_image.save(filename)
        logging.info(f"Saved image to {filename}")
    return canvas

def display_info(frame, processing_time):
    """Display information on the frame.

    Args:
        frame (ndarray): Frame from the video stream.
        processing_time (float): Processing time in secounds.
    """
    FONT_SCALE = 0.0005  # Adjust for larger font size in all images
    THICKNESS_SCALE = 0.001  # Adjust for larger thickness in all images
    height, width = frame.shape[:2]
    cv2.putText(
        frame,
        f"Processing time: {processing_time * 1000:.2f} ms",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=min(width, height) * FONT_SCALE,
        thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
        color=(0, 0, 255),
    )