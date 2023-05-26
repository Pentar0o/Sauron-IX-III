import cv2

import numpy as np

import torch
from torchvision import transforms

from ultralytics import YOLO

from PIL import Image
from PIL import ImageDraw, ImageFont

import urllib.parse
import urllib.request

import os
import json
import time
import io
import sys
import argparse

import colorsys


# classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}


def ValueToRgb(data):
    # Normalize data to the range [0, 1000]
    data = min(max(data, 0), 1000)

    # Calculate the hue and value
    hue = data / 1000
    value = 1 - data / 1000

    # Convert the HSV color to RGB
    R, G, B = colorsys.hsv_to_rgb(hue, 1, value)
    R = int(R * 255)
    G = int(G * 255)
    B = int(B * 255)

    return B, G, R


# Recupération de la liste des caméras dans un fichier JSON
def getListCam(path):
    if os.path.exists(path):
        with open(path,"r") as f:
            listCam = json.load(f)
            if listCam.count !=0:
                return listCam
            else:
                print("File Empty")
                sys.exit(-1)
    else:
        print("File Not Found")
        sys.exit(-1)


# A terme ajouter l'image similarity pour s'assurer qu'il s'agit bien du même véhicule
def GetCCTV(IP, Login, Password, NumCam):
    #Gestion de l'authentification de la caméra
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    top_level_url = "http://" + IP + "/axis-cgi/jpg/image.cgi?camera=" + str(NumCam)
    password_mgr.add_password(None, top_level_url, Login, Password)
    
    handler = urllib.request.HTTPDigestAuthHandler(password_mgr)
    opener = urllib.request.build_opener(handler)
    ImageCCTV = opener.open(top_level_url)
    Contenu = ImageCCTV.read()

    #On lit l'image pour la prédiction
    image = Image.open(io.BytesIO(Contenu))

    return image


def detect_cars(image, model, confidence_threshold, device, cam_index):
    # Pass the image through the YOLO model and use the specified device for inference
    results = model(image, device=device, verbose=False)
    cars = []
    for result in results:
        # Loop through all detections found by YOLOv8
        # Check if at least one detection has a confidence rate higher than the specified threshold
        if any((conf > confidence_threshold) and (model.names[int(cls)] == "car") for conf, cls in zip(result.boxes.conf, result.boxes.cls)):
            for box in result.boxes:
                if box.conf > confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0]

                    # On découpe l'image pour garder que le véhicule
                    CropBox = (int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item()))
                    ImageToCrop = image
                    ImageCopy = ImageToCrop.crop(CropBox).copy()
                    ImageCopy_Tensored = image_to_tensor(ImageCopy, device)
                    
                    cars.append({
                        'id': len(cars) + 1,
                        'x': int(x1),
                        'y': int(y1),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'cam_index': cam_index,
                        'stationary_frames': 0,
                        'tensor': ImageCopy_Tensored
                    })

    return cars


def SaveImage(images):
    # Resize the images to have the same height.
    max_height = max(image.size[1] for image in images)
    resized_images = []
    for image in images:
        height_ratio = max_height / image.size[1]
        new_width = int(image.size[0] * height_ratio)
        resized_image = image.resize((new_width, max_height), Image.LANCZOS)
        resized_images.append(resized_image)

    # Combine the resized images from all cameras into a single image.
    combined_image_array = np.hstack([np.array(image) for image in resized_images])
    combined_image = Image.fromarray(combined_image_array)
    filename = 'Parking.jpg'
    combined_image.save(filename)

    return combined_image_array


def image_similarity(image1_tensor, image2_tensor):
    # Flatten tensors
    image1_tensor = image1_tensor.flatten()
    image2_tensor = image2_tensor.flatten()
    # Calculate cosine similarity
    cos_sim_value = torch.nn.functional.cosine_similarity(image1_tensor, image2_tensor, dim=0).item()
    return cos_sim_value


def image_to_tensor(image, device):
    # Check if image is grayscale
    if image.mode != "L":
        image = image.convert("L")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # Apply transformations to image
    image_tensor = transform(image)

    # Move tensor to device
    image_tensor = image_tensor.to(device)

    # Check tensor shape
    assert image_tensor.shape == (1, 224, 224), f"Unexpected shape for image_tensor: {image_tensor.shape}"

    return image_tensor


def DrawBoundingBoxes(previous_detections, image, font):
    for _, car in previous_detections:
        x1 = car['x']
        y1 = car['y']
        x2 = x1 + car['width']
        y2 = y1 + car['height']

        R, G, B = ValueToRgb(car['stationary_frames'])

        draw = ImageDraw.Draw(image)
        draw.rectangle([(x1, y1), (x2, y2)], outline=(R, G, B), width=5)
        draw.text((x1, y1 - 20), f"ID:{car['id']} / {car['stationary_frames']} frames", fill='red', font=font)

    return image


def compute_centroids(car):
    """Compute the centroids of detected cars."""
    x_max = car['x'] + car['width']
    x_min = car['x']
    y_max = car['y'] +  car['height']
    y_min = car['y']
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    return x_mid, y_mid


def main(model_path, confidence, device, time_between_analyses, trigger):
    # Load camera configuration
    camera_list = getListCam('conf/Camera.json')
    
    # Load the pre-trained YOLO model
    model = YOLO(args.model_path)

    # Set the font for displaying the time above the rectangle
    font = ImageFont.truetype('arial.ttf', 24)

    # Initialize arrays to store detected cars and previous detections for each camera
    for camera in camera_list:
        nb_cam = 5 if camera["Quad"] else 2

        detected_cars = [[] for _ in range(nb_cam)]
        previous_detections = [{} for _ in range(nb_cam)]

        # Initialize the detection array with cars at the moment
        for cam_index in range(1, nb_cam):
            image = GetCCTV(camera["IP"], camera["Login"], camera["Password"], cam_index)
            cars = detect_cars(image, model, args.confidence, args.device, cam_index)
            previous_detections[cam_index] = {car['id']: car for car in cars}

    iteration = 0
    stationary_frames_count = {}
    while True:
        images = []
        
        start_time = time.time()

        for cam_index in range(1, nb_cam):
                image = GetCCTV(camera["IP"], camera["Login"], camera["Password"], cam_index)
                cars = detect_cars(image, model, args.confidence, args.device, cam_index)

                # Update the detected_cars array for this camera
                detected_cars[cam_index] = cars

                # Compare the detected cars with the previous detections for this camera
                for car in detected_cars[cam_index]:
                    if car['id'] in previous_detections[cam_index]:
                        previous = previous_detections[cam_index][car['id']]

                        cos_sim_value = image_similarity(car['tensor'], previous['tensor'])

                        x_car, y_car = compute_centroids(car)
                        x_previous, y_previous = compute_centroids(previous)
                        translation = np.array([x_car - x_previous, y_car - y_previous])
                        distance = np.linalg.norm(translation)

                        if cos_sim_value > trigger or distance < 10 :
                            previous_detections[cam_index][car['id']]['stationary_frames'] += 1
                            if iteration % 10 == 0:
                                previous_detections[cam_index][car['id']]['tensor'] = car['tensor']
                            #print(f"Car OK : {previous_detections[cam_index][car['id']]['id']} - Distance : {distance} / CosSimValue : {cos_sim_value}")
                        else:
                            previous_detections[cam_index][car['id']]['stationary_frames'] -= 1
                            #print(f"Car NOK : {previous_detections[cam_index][car['id']]['id']} - Distance : {distance} / CosSimValue : {cos_sim_value}")
                    else:
                        # No element found with the corresponding id
                        # Add to previous_detections the contents of car
                        previous_detections[cam_index][car['id']] = car

                image = DrawBoundingBoxes(previous_detections[cam_index].items(), image, font)

                images.append(image)


        # Remove an element if after 10 iterations the number of frames has not changed
        if iteration % 10 == 0:
            for cam_index in range(1, nb_cam):
                for car_id in list(previous_detections[cam_index].keys()):
                    if car_id not in stationary_frames_count:
                        stationary_frames_count[car_id] = previous_detections[cam_index][car_id]['stationary_frames']
                    elif stationary_frames_count[car_id] == previous_detections[cam_index][car_id]['stationary_frames']:
                        del previous_detections[cam_index][car_id]
                        del stationary_frames_count[car_id]
                    else:
                        # Replace the comparison photo with the new one after 10 positive iterations
                        stationary_frames_count[car_id] = previous_detections[cam_index][car_id]['stationary_frames']

        iteration += 1

        print("Reset")

        ImageFinale = SaveImage(images)

        # Compute and display processing time in milliseconds
        processing_time_ms = (time.time() - start_time) * 1000
        cv2.putText(ImageFinale,
                    f'Processing time: {processing_time_ms:.2f} ms',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    thickness=2)

        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(ImageFinale, cv2.COLOR_BGR2RGB)

        # Display the result
        cv2.imshow("Détection Voiture Tampon", image_rgb)

        # Exit if the ESC key is pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        cv2.destroyAllWindows()
        time.sleep(args.time_between_analyses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='The path to the YOLO model')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='The confidence threshold for car detection')
    parser.add_argument('--device', default='cpu', help='The device to use for inference')
    parser.add_argument('--time_between_analyses', type=int, default=3600,
                        help='The time between analyses in seconds')
    parser.add_argument('--trigger', type=float, default=0.7,
                        help='The car difference tolerated')
    args = parser.parse_args()

    main(args.model_path, args.confidence, args.device, args.time_between_analyses, args.trigger)
