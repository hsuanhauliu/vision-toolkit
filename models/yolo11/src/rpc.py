# Definitions of RPC request and response data.

from pydantic import BaseModel
from typing import List, Union


class Coordinate(BaseModel):
    """Bounding box coordinate"""
    left: int
    top: int
    right: int
    bottom: int


class ModelInfo(BaseModel):
    """Model information"""
    labels: List[str]
    layers: Union[int, None]
    parameters: Union[int, None]


########## Object Detection ##########
class ObjectDetectionRequest(BaseModel):
    base64_imgs: List[str]  # base64 encoded images


class ObjectDetectionResponse(BaseModel):
    """
    Inference response containing the results.

    The order of both lists align with each other.
    """
    bounding_boxes: List[List[Coordinate]]
    classes: List[List[str]]
    model_info: ModelInfo
