import rpc
import convert

import os
import logging
from typing import Callable, List, Tuple, Union

import numpy as np
from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)

saved_model = "./data/" + os.environ.get("SAVED_MODEL", "yolo_model.pt")
LOGGER.info(f"Loading {saved_model}...")
MODEL = YOLO(saved_model)
LABELS = [name for _, name in MODEL.names.items()]


###### Helper Functions ######
def get_model_info() -> dict:
    model_info = MODEL.info(detailed=False, verbose=True)
    model_layers = model_info[0] if model_info and len(model_info) > 1 else None
    model_parameters = model_info[1] if model_info and len(model_info) > 1 else None
    return {
        "labels": LABELS,
        "layers": model_layers,
        "parameters": model_parameters,
    }


########## Factory Functions ##########
def get_inference_func(task_name: str) -> Callable:
    inference_task = {
        "object_detection": object_detection,
        "image_classification": image_classification,
        "instance_segmentation": instance_segmentation,
        "pose_estimation": pose_estimation,
        "oriented_object_detection": oriented_object_detection,
    }

    if task_name not in inference_task:
        raise ValueError(f"Unsupported inference task: '{task_name}'. ")
    return inference_task[task_name]


def get_response_model(task_name: str) -> Callable:
    response_model = {
        "object_detection": rpc.ObjectDetectionResponse,
        "image_classification": rpc.ImageClassificationResponse,
        "instance_segmentation": rpc.InstanceSegmentationResponse,
        "pose_estimation": rpc.PoseEstimationResponse,
        "oriented_object_detection": rpc.OrientedObjectDetectionResponse,
    }

    if task_name not in response_model:
        raise ValueError(f"Unsupported response model: '{task_name}'. ")
    return response_model[task_name]


########## Inference functions ##########
def object_detection(data: rpc.ObjectDetectionRequest):
    """Object detection feature entry."""
    imgs = convert.base64ListToNumpyArrayList(data.base64_imgs)
    results = MODEL(imgs)
    bb, classes = convert.detection_results_to_lists(results, MODEL.names)

    return {
        "bounding_boxes": bb,
        "classes": classes,
        "model_info": get_model_info(),
    }

def oriented_object_detection(data: rpc.OrientedObjectDetectionRequest):
    """Object detection feature entry."""
    imgs = convert.base64ListToNumpyArrayList(data.base64_imgs)
    results = MODEL(imgs)
    bb, classes = convert.oriented_detection_results_to_lists(results, MODEL.names)

    return {
        "oriented_bounding_boxes": bb,
        "classes": classes,
        "model_info": get_model_info(),
    }


def image_classification(data: rpc.ImageClassificationRequest):
    """Image classification feature entry."""
    imgs = convert.base64ListToNumpyArrayList(data.base64_imgs)
    results = MODEL(imgs)
    classes, probs = convert.classification_results_to_lists(results, MODEL.names)

    return {
        "classes": classes,
        "probabilities": probs,
        "model_info": get_model_info(),
    }


def instance_segmentation(data: rpc.InstanceSegmentationRequest):
    """Instance segmentation feature entry."""
    imgs = convert.base64ListToNumpyArrayList(data.base64_imgs)
    results = MODEL(imgs)
    polygons, classes = convert.segmentation_results_to_lists(results, MODEL.names)

    return {
        "polygons": polygons,
        "classes": classes,
        "model_info": get_model_info(),
    }


def pose_estimation(data: rpc.PoseEstimationRequest):
    """Pose estimation feature entry."""
    imgs = convert.base64ListToNumpyArrayList(data.base64_imgs)
    results = MODEL(imgs)
    keypoints, classes = convert.estimation_results_to_lists(results, MODEL.names)

    return {
        "keypoints": keypoints,
        "classes": classes,
        "model_info": get_model_info(),
    }
