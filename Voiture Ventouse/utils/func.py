import math
from datetime import datetime
from io import StringIO

import sys
import os
import json
import boto3
import numpy as np
import pymsteams
import torch
from torchvision import transforms
from ultralytics import YOLO

def get_list_cam(path: str) -> list:
    """ Parse a JSON file to get the list of cameras
        exit if the file is empty or not found

    Args:
        path (str): Path to the JSON file

    Returns:
        list: List of cameras
    """    
    if os.path.exists(path):
        with open(path,"r") as f:
            list_cam = json.load(f)
            if len(list_cam) == 0:
                print("Cameras not found")
                sys.exit(-1)
            return list_cam

    else:
        print("File Not Found")
        sys.exit(-1)


def test_credentials(config: dict):
    """Test the credentials of the AWS account."""
    client = boto3.client('s3', aws_access_key_id=config["aws_access_key"], aws_secret_access_key=config["aws_secret_key"])
    response = client.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    if not config["aws_bucket_name"] in buckets:
        raise Exception(f"Le bucket {config['aws_bucket_name']} n'existe pas.")    

def send_xml_s3(bndbox, size, path, config):
    """Envoi des données sur S3"""
    session = boto3.Session(
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key
    )
    filename = path.split("/")[-1]
    data = create_xml(bndbox, filename, size[1], size[0], "truck")
    # Creating S3 Resource From the Session.
    s3_ressource = session.resource("s3")

    s3_object = s3_ressource.Object(config.bucket_name, path) # type: ignore
    s3_object.put(Body=data)

def send_image_s3(data, path, nom_camera, config, webhook_teams):
    """Envoi des données sur S3"""
    session = boto3.Session(
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key
    )

    # Creating S3 Resource From the Session.
    s3_ressource = session.resource("s3")

    s3_object = s3_ressource.Object(config.bucket_name, path) # type: ignore
    s3_object.put(Body=data)
    object_key = s3_object.key

    send_to_teams(s3_ressource, object_key, nom_camera, config, webhook_teams)


def send_to_teams(s3_ressource, object_key, nom_camera, config, webhook_url):
    now = datetime.now()
    formatted_date = now.strftime('%d-%m-%Y %H:%M')

    try :
        # Générer une URL de partage
        url = s3_ressource.meta.client.generate_presigned_url(
            'get_object',
            Params={'Bucket': config.bucket_name, 'Key': object_key},
            ExpiresIn=36000  # Durée de validité de l'URL en secondes
        )

        myTeamsMessage = pymsteams.connectorcard(webhook_url)

        myTeamsMessage.title(f"Détection d'un 3.5t via la caméra {nom_camera} : {formatted_date}")
        myTeamsMessage.text(url)
        myTeamsMessage.send()
    except Exception as e :
        print(f"Erreur : {e}")
        pass


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


def stationnement_jours(nb_jours):
    if nb_jours <= 0:
        return None

    periode = 2 * math.pi / nb_jours
    couleur_premier_jour: tuple[int,int,int] = (0, 255, 255)

    couleurs: list[tuple[int,int,int]] = [couleur_premier_jour]

    for jour in range(2, nb_jours):
        angle = (jour - 1) * periode + math.pi
        rouge = int(math.cos(angle) * 127 + 128)
        vert = int(math.cos(angle + 2 * math.pi / 3) * 127 + 128)
        bleu = int(math.cos(angle + 4 * math.pi / 3) * 127 + 128)

        # Pour les 10 derniers pourcentages de jours, pencher vers une couleur sombre
        if jour > nb_jours * 0.9:
            facteur_sombre = 1 - ((jour - nb_jours * 0.9) / (nb_jours * 0.1))
            sombre = 50  # Intensité de couleur sombre (gris foncé)
            rouge = int(rouge * facteur_sombre + sombre * (1 - facteur_sombre))
            vert = int(vert * facteur_sombre + sombre * (1 - facteur_sombre))
            bleu = int(bleu * facteur_sombre + sombre * (1 - facteur_sombre))

        nouvelle_couleur = (rouge, vert, bleu)
        couleurs.append(nouvelle_couleur)

    couleur_dernier_jour = (0, 0, 0)
    couleurs.append(couleur_dernier_jour)

    return couleurs


def calculate_bbox_area(box):
    """
        Calculate the area of a yolo bbox
    """
    return (box[2] - box[0]) * (box[3] - box[1])

def create_xml(bdnbx, filename, width, height, classe):
    xmin = bdnbx[0]
    ymin = bdnbx[1]
    xmax = bdnbx[2]
    ymax = bdnbx[3]
    fichier_xml = StringIO()
    fichier_xml.write(f"<annotation verified=\"yes\"><filename>{filename}</filename><size><width>{width}</width><height>{height}</height><depth>3</depth></size>")
    fichier_xml.write(f"<object><name>{classe}</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object>")
    fichier_xml.write("</annotation>")
    xml_data = fichier_xml.getvalue().encode("utf-8")
    fichier_xml.close()
    return xml_data


def image_similarity(image1_tensor: np.ndarray, image2_tensor: np.ndarray) -> float:
    """Calculate cosine similarity between two images.

    Args:
        image1_tensor (ndarray): The first image tensor.
        image2_tensor (ndarray): The second image tensor.

    Returns:
        float: The cosine similarity value.
    """
    # Flatten tensors
    image1_tensor = image1_tensor.flatten()
    image2_tensor = image2_tensor.flatten()
    # Calculate cosine similarity
    cos_sim_value = torch.nn.functional.cosine_similarity(
        image1_tensor, image2_tensor, dim=0 # type: ignore
    ).item()
    return cos_sim_value

def calculate_iou(bbox1: tuple[int,int,int,int], bbox2: tuple[int,int,int,int]):
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

