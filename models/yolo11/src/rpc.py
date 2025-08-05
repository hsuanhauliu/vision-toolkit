# Definitions of data request and response data.

from pydantic import BaseModel
from typing import List

import data



########## Object Detection ##########
class ObjectDetectionRequest(BaseModel):
    base64_imgs: List[str]  # base64 encoded images


class ObjectDetectionResponse(BaseModel):
    """
    Inference response containing the results.

    The order of both lists align with each other.
    """

    bounding_boxes: List[List[data.BoundingBox]]
    classes: List[List[str]]
    model_info: data.ModelInfo


class InstanceSegmentationRequest(BaseModel):
    base64_imgs: List[str]  # base64 encoded images


class InstanceSegmentationResponse(BaseModel):
    polygons: List[List[data.Polygon]]
    classes: List[List[str]]
    model_info: data.ModelInfo
