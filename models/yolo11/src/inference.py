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
def detection_results_to_lists(model_results, class_names):
    """
    Convert the model inference results to a list of bounding boxes and a list of corresponding class names.
    """
    all_bb, all_classes = [], []
    for result in model_results:
        bb, classes = [], []
        for box in result.boxes:
            # get box coordinates in (left, top, right, bottom)
            bb.append(convert.boxToCoordinates(box))
            classes.append(class_names[int(box.cls)])
        all_bb.append(bb)
        all_classes.append(classes)
    return all_bb, all_classes


def segmentation_results_to_lists(model_results, class_names):
    all_polygons, all_classes = [], []
    for result in model_results:
        masks, classes = [], []
        for i, mask in enumerate(result.masks):
            classes.append(class_names[int(result.boxes.cls[i])])
            masks.append(convert.maskToPolygon(mask))
        all_polygons.append(masks)
        all_classes.append(classes)
    return all_polygons, all_classes


def estimation_results_to_lists(model_results, class_names):
    all_keypoints, all_classes = [], []
    for result in model_results:
        keypoints, classes = [], []
        for i, keypoint in enumerate(result.keypoints):
            classes.append(class_names[int(result.boxes.cls[i])])
            keypoints.append(convert.keypointsToHumanBodyKeypoints(keypoint))
        all_keypoints.append(keypoints)
        all_classes.append(classes)
    return all_keypoints, all_classes


def get_model_info() -> dict:
    model_info = MODEL.info(detailed=False, verbose=True)
    model_layers = model_info[0] if model_info and len(model_info) > 1 else None
    model_parameters = model_info[1] if model_info and len(model_info) > 1 else None
    return {
        "labels": LABELS,
        "layers": model_layers,
        "parameters": model_parameters,
    }


def get_inference_func(task_name: str) -> Callable:
    inference_task = {
        "object_detection": object_detection,
        "instance_segmentation": instance_segmentation,
        "pose_estimation": pose_estimation,
    }

    if task_name not in inference_task:
        raise ValueError(f"Unsupported inference task: '{task_name}'. ")
    return inference_task[task_name]


def get_response_model(task_name: str) -> Callable:
    response_model = {
        "object_detection": rpc.ObjectDetectionResponse,
        "instance_segmentation": rpc.InstanceSegmentationResponse,
        "pose_estimation": rpc.PoseEstimationResponse,
    }

    if task_name not in response_model:
        raise ValueError(f"Unsupported response model: '{task_name}'. ")
    return response_model[task_name]


########## Inference functions ##########
def object_detection(data: rpc.ObjectDetectionRequest):
    """Object detection feature entry."""
    imgs = convert.base64ListToNumpyArrayList(data.base64_imgs)
    results = MODEL(imgs)
    bb, classes = detection_results_to_lists(results, MODEL.names)

    return {
        "bounding_boxes": bb,
        "classes": classes,
        "model_info": get_model_info(),
    }


def instance_segmentation(data: rpc.InstanceSegmentationRequest):
    """Instance segmentation feature entry."""
    imgs = convert.base64ListToNumpyArrayList(data.base64_imgs)
    results = MODEL(imgs)
    polygons, classes = segmentation_results_to_lists(results, MODEL.names)

    return {
        "polygons": polygons,
        "classes": classes,
        "model_info": get_model_info(),
    }


def pose_estimation(data: rpc.PoseEstimationRequest):
    """Pose estimation feature entry."""
    imgs = convert.base64ListToNumpyArrayList(data.base64_imgs)
    results = MODEL(imgs)
    keypoints, classes = estimation_results_to_lists(results, MODEL.names)

    return {
        "keypoints": keypoints,
        "classes": classes,
        "model_info": get_model_info(),
    }
