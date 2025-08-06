from pydantic import BaseModel
from typing import List, Union


class BoundingBox(BaseModel):
    """Bounding box coordinate in pixel values"""

    left: float
    top: float
    right: float
    bottom: float


class OrientedBoundingBox(BaseModel):
    """Oriented bounding box coordinate in pixel values"""

    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float


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
