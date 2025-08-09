# Run this script from the parent directory
# python ./src/inference_img.py

import cv2
import numpy as np
from pydantic import BaseModel, Field
from ultralytics import SAM

import os
import base64
import logging
from typing import List, Union, Optional

from fast_serve import create_app
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

LOGGER = logging.getLogger(__name__)
saved_model = "./data/" + os.environ.get("SAVED_MODEL", "yolo_model.pt")
LOGGER.info(f"Loading {saved_model}...")
MODEL = SAM(saved_model)


########## Helpers ##########
def base64ToNumpyArray(img_str: str) -> np.ndarray:
    """Convert base64 image to numpy array."""
    if "," in img_str:
        # Split the string by the comma and take the second part
        img_str = img_str.split(",")[1]

    img_bytes = base64.b64decode(img_str)
    img = np.fromstring(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)


def base64ListToNumpyArrayList(img_strs: List[str]) -> List[np.ndarray]:
    imgs = []
    for img in img_strs:
        imgs.append(base64ToNumpyArray(img))
    return imgs


def get_model_info() -> dict:
    model_info = MODEL.info(detailed=False, verbose=True)
    model_layers = model_info[0] if model_info and len(model_info) > 1 else None
    model_parameters = model_info[1] if model_info and len(model_info) > 1 else None
    return {
        "layers": model_layers,
        "parameters": model_parameters,
    }


def maskToBase64(mask) -> str:
    np_array = mask.data[0].byte().numpy() * 255
    success, buffer = cv2.imencode(".png", np_array)
    if not success:
        raise Exception("Could not encode image with OpenCV.")

    img_bytes = buffer.tobytes()
    return base64.b64encode(img_bytes).decode("utf-8")


def extractPoints(data):
    points, labels = [], []
    for points_per_img in data.points:
        points.append([[point.x, point.y] for point in points_per_img])
        labels.append([int(point.label) for point in points_per_img])
    return points, labels


def extractBoundingBoxes(data):
    bboxes = []
    for bboxes_per_img in data.bounding_boxes:
        bboxes.append(
            [[bbox.left, bbox.top, bbox.right, bbox.bottom] for bbox in bboxes_per_img]
        )
    return bboxes


########## Data Abstractions ##########
class ModelInfo(BaseModel):
    """Model information"""

    layers: Union[int, None]
    parameters: Union[int, None]


class Point(BaseModel):
    """Coordinates in pixel values"""

    x: int
    y: int
    label: bool  # true=1, false=0


class BoundingBox(BaseModel):
    """Bounding box coordinate in pixel values"""

    left: int
    top: int
    right: int
    bottom: int


class Sam2Request(BaseModel):
    """Request data for SAM2."""

    base64_imgs: List[str]  # base64 encoded images

    # User prompts one-of. Priority:
    # 1. everything
    # 2. points
    # 3. bounding_boxes
    everything: Optional[bool] = Field(None)
    points: Optional[List[List[Point]]] = Field(None)
    bounding_boxes: Optional[List[List[BoundingBox]]] = Field(None)


class Sam2Response(BaseModel):
    """Response data for SAM2."""

    masks: List[List[str]]  # base64 encoded binary mask
    model_info: ModelInfo


########## Inference Function ##########
def instance_segmentation(data: Sam2Request):
    imgs = base64ListToNumpyArrayList(data.base64_imgs)
    model_results = None
    if data.everything:
        model_results = MODEL(imgs)
    elif data.points:
        points, labels = extractPoints(data)
        model_results = MODEL(imgs, points=points, labels=labels)
    elif data.bounding_boxes:
        bboxes = extractBoundingBoxes(data)
        model_results = MODEL(imgs, bboxes=bboxes)
    else:
        raise Exception("Invalid model inference mode")

    all_masks = []
    for result in model_results:
        masks = []
        for _, mask in enumerate(result.masks):
            masks.append(maskToBase64(mask))
        all_masks.append(masks)

    return {
        "masks": all_masks,
        "model_info": get_model_info(),
    }


########## API ##########
LOGGER.info(f"Preparing API endpoints for SAM2 model")
app = create_app(instance_segmentation, response_model=Sam2Response)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "null",  # Allow requests from file:// protocol
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, including OPTIONS
    allow_headers=["*"],  # Allow all headers
)
