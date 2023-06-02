""" parked_car_tracker.py
    Track cars with video streams from IP cameras.
    Using YOLOv8 to detect cars and a deep sort tracker to track them.
    Usage:
        python parked_car_tracker.py <model_path> [--confidence <confidence>] [--device <device>] [--interval <interval>]
    Arguments:
        model_path  - The path to the YOLO model.
    Options:
        --confidence <confidence>  - The confidence threshold for car detection [default: 0.5].
        --device <device>          - The device to use for inference [default: cpu].
        --interval <interval>      - The time between analyses in seconds [default: 3600].
        --time <time>              - The time in seconds after which the car is not anymore in the parking [default: 10].
"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from ultralytics import YOLO

from utils import Camera

def argument_parser():
    """Parse command line arguments.
    Returns:
        (dict): command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='The path to the YOLO model')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='The confidence threshold for car detection')
    parser.add_argument('--device', default='cpu', help='The device to use for inference')
    parser.add_argument('--interval', type=int, default=3600,
                        help='The time between analyses in seconds')
    
    parser.add_argument('--time', type=int, default=10,
                    help='The time after which the car is not anymore in the parking')
    
    parser.add_argument('--trigger', type=float, default=0.7,
                        help='The car difference tolerated')
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    return args

def load_model(model_path):
    """Load the YOLO model.

    Args:
        model_path (str): Path to the YOLO model.

    Returns:
        model (YOLO): YOLO model.
    """
    model = YOLO(model_path)
    return model


def image_to_tensor(image, device):
    """Convert image to tensor.

    Args:
        image (PIL.Image): The image to be converted.
        device (torch.device): The device to move tensor to.

    Returns:
        torch.Tensor: The converted tensor.
    """

    # Check if image is grayscale
    if image.mode != "L":
        image = image.convert("L")

    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )

    # Apply transformations to image
    image_tensor = transform(image)

    # Move tensor to device
    image_tensor = image_tensor.to(device)

    # Check tensor shape
    assert image_tensor.shape == (
        1,
        224,
        224,
    ), f"Unexpected shape for image_tensor: {image_tensor.shape}"

    return image_tensor


def detect_cars(model, frame, confidence, classes, cam_index):
    """Detect cars in a frame using YOLOv8.

    Args:
        model (YOLO): YOLO model.
        frame (ndarray): Frame from the video stream.
        confidence (float): Confidence threshold for car detection.
        classes (list): List of classes to detect.
        cam_index (int): Camera index.

    Returns:
        tracks (list): List of tracks.
    """

    tracks = []
    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform car detection using Ultralytics
    results = model(rgb_frame, device=DEVICE, verbose=False)
    # Extract car / bus / truck detections with confidence > 0.5
    for result in results:
        if any((conf > confidence) and (model.names[int(cls)] == "voiture") for conf, cls in zip(result.boxes.conf, result.boxes.cls)):
        # Loop through all detections found by YOLOv8
            for box in result.boxes:
                if (int(box.cls) in classes) and box.conf > confidence:
                    start_x, start_y, end_x, end_y  = box.xyxy[0]
            
                    # crop the car
                    car = frame[int(start_y):int(end_y), int(start_x):int(end_x)]
                    # convert the car to tensor
                    tensor = image_to_tensor(Image.fromarray(car), DEVICE)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    tracks.append({
                        "bbox": (int(start_x), int(start_y), int(end_x), int(end_y)),
                        "cam_index": cam_index,
                        "tensor": tensor,
                    })
    return tracks


def box_label(image, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
    """Draw a bounding box with a label on an image.

    Args:
        image (ndarray): Image to draw on.
        box (tuple): Bounding box coordinates.
        label (str, optional): Label to draw. Defaults to ''.
        color (tuple, optional): Color of the bounding box. Defaults to (128, 128, 128).
        txt_color (tuple, optional): Color of the label. Defaults to (255, 255, 255).
    """
    line_width = max(round(sum(image.shape) / 3 * 0.003), 2)
    left_top, right_bottom = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(
        image, left_top, right_bottom, color, thickness=line_width, lineType=cv2.LINE_AA
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
        cv2.rectangle(image, left_top, right_bottom, color, -1, cv2.LINE_AA)  # filled
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


def draw_tracks(frame, cars):
    """Draw bounding boxes and directions of tracked cars.

    Args:
        frame (ndarray): Frame from the video stream.
        cars (OrderedDict): List of tracked cars.
    """
    # loop over the tracked objects
    for object_id, car in cars.items():
        rect = car.original_rect
        box_label(frame, rect, label=f"Car {object_id}", color=car.color())


# Recupération de la liste des caméras dans un fichier JSON
def get_list_cam(path):
    """ Parse a JSON file to get the list of cameras
        exit if the file is empty or not found

    Args:
        path (str): Path to the JSON file

    Returns:
        list_cam: List of cameras
    """    
    if os.path.exists(path):
        with open(path,"r") as f:
            list_cam = json.load(f)
            if list_cam.count !=0:
                return list_cam
            else:
                print("File Empty")
                sys.exit(-1)
    else:
        print("File Not Found")
        sys.exit(-1)


def save_image(images, camera):
    """Save the images from the cameras to a single image.

    Args:
        images (list): List of images from the cameras.
        camera (Camera): Camera object.
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
    
    grid_image = Image.fromarray(canvas)
    camera_name = camera.name + "_Quad" if camera.quad else ""
    filename = f'{camera_name}-Parking.jpg'
    grid_image.save(filename)
    logging.info(f"Saved image to {filename}")
    return grid_image

def main():
    """Main function."""
    args = argument_parser()
    logging.debug("Starting the application...")
    # Load the YOLO model
    logging.debug("Loading the YOLO model...")
    model = load_model(args.model_path)
    global DEVICE 
    DEVICE = args.device
    if model.names is None:
        logging.error("Could not load the YOLO labels.")
        sys.exit(1)
    classes = dict(
        filter(
            lambda x: x[1] in ["car", "voiture", "bus", "truck"], model.names.items()
        )
    )
    classes = list(classes.keys())
    logging.debug(classes)
    # Create a VideoCapture object using the RTSP URL
    logging.debug("Connecting to video stream...")
    
    
    cameras = list(map(lambda x: Camera(x, args.time), get_list_cam("conf/Camera.json")))
    for camera in cameras:
        camera.wait_until_ready()

    # Set coordinates of virtual line
    # line = (1300, 300, 950, 500)
    try:
        while True:
            for camera in cameras:
                images = []
                nb_cam = 4 if camera.quad else 1
                for cam_index in range(nb_cam):                    
                    # Read a new frame from the video stream
                    frame, centroid_tracker = camera.get_frame(cam_index)
                    if frame is not None and centroid_tracker is not None:
                
                        tracks = detect_cars(model, frame, args.confidence, classes, cam_index + 1)
                        rects = [t["bbox"] for t in tracks]
                        cars_image = [t['tensor'] for t in tracks]
                        cars = centroid_tracker.update(rects, cars_image)
                        centroid_tracker.delete_double(0.8)
                        draw_tracks(frame, cars)
                    else:
                        sys.exit(255)
                        break
                    # Draw the virtual line on each frame
                    # cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), thickness=3) r
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    images.append(Image.fromarray(frame))
                save_image(images, camera)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Keyboard interrupt")

    finally:
        for camera in cameras:
            camera.stop()


if __name__ == "__main__":
    main()
