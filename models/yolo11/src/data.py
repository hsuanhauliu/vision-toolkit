from pydantic import BaseModel
from typing import List, Union


class BoundingBox(BaseModel):
    """Bounding box coordinate in pixel values"""

    left: int
    top: int
    right: int
    bottom: int


class Coordinate(BaseModel):
    """Coordinates in pixel values"""

    x: float
    y: float


class Polygon(BaseModel):
    """Polygon/contour of the mask"""

    contour: List[Coordinate]


class HumanBodyKeypoints(BaseModel):
    """Coco dataset defined human body keypoints."""

    nose: Coordinate
    left_eye: Coordinate
    right_eye: Coordinate
    left_ear: Coordinate
    right_ear: Coordinate
    left_shoulder: Coordinate
    right_shoulder: Coordinate
    left_elbow: Coordinate
    right_elbow: Coordinate
    left_wrist: Coordinate
    right_wrist: Coordinate
    left_hip: Coordinate
    right_hip: Coordinate
    left_knee: Coordinate
    right_knee: Coordinate
    left_ankle: Coordinate
    right_ankle: Coordinate


class ModelInfo(BaseModel):
    """Model information"""

    labels: List[str]
    layers: Union[int, None]
    parameters: Union[int, None]
