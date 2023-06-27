""" parked_car_tracker.py
    Track cars with video streams from IP cameras.
    Using YOLOv8 to detect cars and a deep sort tracker to track them.
    Usage:
        python parked_car_tracker.py <model_path> [--confidence <confidence>] [--device <device>] [--interval <interval>] [--time <time>] [--iou_threshold <iou_threshold>] 
    Arguments:
        model_path  - The path to the YOLO model.
    Options:
        --confidence <confidence>  - The confidence threshold for car detection [default: 0.5].
        --device <device>          - The device to use for inference [default: cpu].
        --interval <interval>      - The time between analyses in seconds [default: 3600].
        --time <time>              - The time in seconds after which the car is not anymore in the parking [default: 10].
        --iou_threshold <iou_threshold>  - The iou threshold for removing duplicate detections [default: 0.7].
        --similarity_threshold <similarity_threshold>  - The similarity threshold for considering two cars as the same [default: 0.7].
        --display                  - Display the analysis.
        --save                     - Save the analysis.
        --debug                                         - Print lots of debugging statements.
        --verbose                                       - Be verbose.
        -d, --display                                   - Display the analysis.
        -h --help                  - Show this screen.
"""

import argparse
import logging
import sys
import time

import cv2
from PIL import Image
from utils import Camera, YoloDetector
from utils.func import *
from utils.draw_helper import make_image, draw_tracks

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
    
    parser.add_argument('--iou_threshold', type=float, default=0.7,
                        help='The car difference tolerated')
    parser.add_argument('--similarity_threshold', type=float, default=0.7,
                        help='The similarity threshold for considering two cars as the same')
    
    parser.add_argument('-d','--display', action='store_true', help='Display the analysis')

    parser.add_argument('--save', action='store_true', help='Save the analysis')

    parser.add_argument(
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    return args

def main():
    """Main function."""
    args = argument_parser()
    logging.debug("Starting the application...")
    # Load the YOLO model
    logging.debug("Loading the YOLO model...")

    yolo_detector = YoloDetector(args.model_path, args.device, args.confidence, ["car", "voiture", "bus", "truck"]) 

    colors = stationnement_jours(7)
    if colors is None:
        logging.error("Could not load the colors.")
        sys.exit(1)
    logging.debug(yolo_detector.classes)
    # Create a VideoCapture object using the RTSP URL
    logging.debug("Connecting to video stream...")
    
    
    cameras = list(map(lambda x: Camera(x, args.time, args.similarity_threshold), get_list_cam("conf/Camera.json")))
    try:
        for camera in cameras:
            camera.wait_until_ready()
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    logging.debug("Connected to video stream.")

    try:
        while True:
            for camera in cameras:
                logging.debug(f"Processing camera {camera.name}...")
                images = []
                nb_cam = 4 if camera.quad else 1
                for cam_index in range(nb_cam):                    
                    # Read a new frame from the video stream
                    frame, centroid_tracker = camera.get_frame(cam_index)
                    if frame is not None and centroid_tracker is not None:
                        tracks = yolo_detector.detect(frame, cam_index + 1)
                        rects = [t["bbox"] for t in tracks]
                        cars_image = [t['tensor'] for t in tracks]
                        cars = centroid_tracker.update(rects, cars_image)
                        centroid_tracker.delete_double(args.iou_threshold)
                        draw_tracks(frame, cars, colors)
                    else:
                        Exception("No frame or centroid tracker")                    
                    images.append(Image.fromarray(frame))

                if args.display:
                    # Display the resulting frame
                    if camera.quad:
                        img = make_image(images, camera, args.save)
                        cv2.imshow(f"{camera.name}", img)
                    else:
                        cv2.imshow(f"{camera.name}", images[0])
                    # Press Q on keyboard to stop recording
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Keyboard interrupt")

    except Exception as e:
        logging.error(e)

    finally:
        for camera in cameras:
            camera.stop()


if __name__ == "__main__":
    main()
