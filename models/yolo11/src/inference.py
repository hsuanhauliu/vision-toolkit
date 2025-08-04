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
def to_lists(
    model_results, class_names
) -> Tuple[List[List[rpc.Coordinate]], List[List[str]]]:
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


def get_inference_func(task_name: str) -> Callable:
    inference_task = {
        "object_detection": object_detection,
    }

    if task_name not in inference_task:
        raise ValueError(f"Unsupported inference task: '{task_name}'. ")
    return inference_task[task_name]


def get_response_model(task_name: str) -> Callable:
    response_model = {
        "object_detection": rpc.ObjectDetectionResponse,
    }

    if task_name not in response_model:
        raise ValueError(f"Unsupported response model: '{task_name}'. ")
    return response_model[task_name]


########## Inference functions ##########
def object_detection(data: rpc.ObjectDetectionRequest) -> dict[str, Union[int, str]]:
    imgs = convert.base64ListToNumpyArrayList(data.base64_imgs)
    results = MODEL(imgs)
    bb, classes = to_lists(results, MODEL.names)

    model_info = MODEL.info(detailed=False, verbose=True)
    model_layers = model_info[0] if model_info and len(model_info) > 1 else None
    model_parameters = model_info[1] if model_info and len(model_info) > 1 else None

    return {
        "bounding_boxes": bb,
        "classes": classes,
        "model_info": {
            "labels": LABELS,
            "layers": model_layers,
            "parameters": model_parameters,
        },
    }


def image_segmentation():
    return


def image_classification():
    return
