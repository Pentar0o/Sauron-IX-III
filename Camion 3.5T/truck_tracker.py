"""
    Track trucks with video streams from IP cameras.\n
    Using YOLOv8 to detect trucks and a sort tracker to track them.\n
    The script will send detections in a S3 bucket [and save them in a local file if needed].\n
    It will also send a message to a Teams channel if a truck is detected.\n\n

    The script assume the presence of a Camera.json file in the conf directory for the configuration of the cameras and Teams webhook.\n
    The script assume the presence of a Auth.json file in the conf directory for the S3 authentication to the S3 bucket.\n\n

    Usage:
        python truck_tracker.py <model_path> <area_threshold> [options]
    Arguments:
        model_path      - The path to the YOLO model.
        area_threshold  - The area of the truck in pixels.
    Options:
        --confidence <confidence>                      - The confidence threshold for object detection [default: 0.5].\n
        --device <device>                              - The device to use for inference [default: cpu].
        --interval <interval>                          - The time between analyses in seconds [default: 3600].
        --time <time>                                  - The time after which the object is not anymore in the cctv in seconds [default: 10].
        --iou_threshold <iou_threshold>                - The object difference tolerated [default: 0.7].
        --similarity_threshold <similarity_threshold>  - The similarity threshold for considering two detections as the same [default: 0.7].
        --division                                      - If we want to divide the image in multiple parts [default: false].
        --debug                                         - Print lots of debugging statements.
        --verbose                                       - Be verbose.
        -d, --display                                   - Display the analysis.
        -s, --save                                      - Save the analysis.
        -h, --help                                      - Show this screen.
"""

import argparse
import logging
import sys
import time

import cv2
from PIL import Image
from utils import Camera, YoloDetector
from utils.func import *
from utils.draw_helper import make_image, display_info, draw_tracks


def argument_parser():
    """Parse command line arguments.
    Returns:
        argparse.ArgumentParser -- Object containing command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='The path to the YOLO model')
    parser.add_argument('area_threshold', type=int,
                        help='The area of the truck')
    
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
    
    parser.add_argument('--division', action='store_true', help='Divide the frame in parts for analysis')

    parser.add_argument('-d','--display', action='store_true', help='Display the analysis')

    parser.add_argument('-s','--save', action='store_true', help='Save the analysis')

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
    aws_conf = parse_aws_json("conf/Auth.json")
    logging.debug("Testing the connection to AWS...")
    try:
        test_credentials(aws_conf)
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    logging.debug("Loading the YOLO model...")

    yolo_detector = YoloDetector(args.model_path, args.device, args.confidence, args.division, args.area_threshold, ["truck","bus"]) 
    logging.debug(yolo_detector.classes)
    # Create a VideoCapture object using the RTSP URL
    logging.debug("Connecting to video stream...")
    webhook_teams, list_cam = get_list_cam("conf/Camera.json")
    cameras = list(map(lambda x: Camera(x, aws_conf, webhook_teams, args.time, args.similarity_threshold), list_cam))
    for camera in cameras:
        camera.wait_until_ready()
    try:
        while True:
            for camera in cameras:
                start_time = time.time()
                images = []
                nb_cam = 4 if camera.quad else 1
                for cam_index in range(nb_cam):                    
                    # Read a new frame from the video stream
                    frame, centroid_tracker = camera.get_frame(cam_index)
                    if frame is not None and centroid_tracker is not None:
                        tracks = yolo_detector.detect(frame, cam_index + 1)
                        centroid_tracker.update(tracks, frame)
                        centroid_tracker.delete_double(args.iou_threshold)
                    else:
                        raise Exception("No frame received")
                    if args.display:
                        draw_tracks(frame, centroid_tracker.objects)
                    images.append(Image.fromarray(frame))

                processing_time = time.time() - start_time
                if args.display:
                    # Display the resulting frame
                    if camera.quad:
                        img = make_image(images, camera, args.save)
                        display_info(img, processing_time)
                        cv2.imshow(f"{camera.name}", img)
                    else:
                        display_info(images[0], processing_time)
                        cv2.imshow(f"{camera.name}", images[0])
                    # Press Q on keyboard to stop recording
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except KeyboardInterrupt:
        print("Keyboard interrupt")
    
    except Exception as e:
        logging.error(e)

    finally:
        for camera in cameras:
            camera.stop()


if __name__ == "__main__":
    main()
