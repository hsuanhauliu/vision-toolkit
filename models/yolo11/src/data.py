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


class ModelInfo(BaseModel):
    """Model information"""

    labels: List[str]
    layers: Union[int, None]
    parameters: Union[int, None]
