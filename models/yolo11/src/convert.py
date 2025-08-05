# Helper functions for type conversion.
import base64

import cv2
import numpy as np
from typing import List

import data


def base64ListToNumpyArrayList(img_strs: List[str]) -> List[np.ndarray]:
    imgs = []
    for img in img_strs:
        imgs.append(base64ToNumpyArray(img))
    return imgs


def base64ToNumpyArray(img_str: str) -> np.ndarray:
    """Convert base64 image to numpy array."""
    if "," in img_str:
        # Split the string by the comma and take the second part
        img_str = img_str.split(",")[1]

    img_bytes = base64.b64decode(img_str)
    img = np.fromstring(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)


def numpyArrayToBase64(img: np.ndarray) -> str:
    return base64.b64encode(img)


def boxToCoordinates(box) -> data.BoundingBox:
    "Convert a Ultralytics Box class to dictionary of coordinates."
    bounding_box = box.xyxy[0].numpy()
    return data.BoundingBox(
        left=int(bounding_box[0]),
        top=int(bounding_box[1]),
        right=int(bounding_box[2]),
        bottom=int(bounding_box[3]),
    )


def maskToPolygon(mask) -> data.Polygon:
    contour = mask.xy[0]

    # deduplicate points
    points = []
    for point in contour:
        coor = (point[0], point[1])
        if not len(points):
            points.append(data.Coordinate(x=coor[0], y=coor[1]))
        elif coor[0] != points[-1].x or coor[1] != points[-1].y:
            points.append(data.Coordinate(x=coor[0], y=coor[1]))
    return data.Polygon(contour=points)
