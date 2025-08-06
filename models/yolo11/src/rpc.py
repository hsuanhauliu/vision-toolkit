# Definitions of data request and response data.

from pydantic import BaseModel
from typing import List

import data


########## Object Detection ##########
class ObjectDetectionRequest(BaseModel):
    """Object detection RPC request."""

    base64_imgs: List[str]  # base64 encoded images


class ObjectDetectionResponse(BaseModel):
    """Object detection RPC response.

    The order of both lists align with each other.
    """

    bounding_boxes: List[List[data.BoundingBox]]
    classes: List[List[str]]
    model_info: data.ModelInfo


########## Instance Segmentation ##########
class InstanceSegmentationRequest(BaseModel):
    """Instance segmentation RPC request."""

    base64_imgs: List[str]  # base64 encoded images


class InstanceSegmentationResponse(BaseModel):
    """Instance segmentation RPC response.

    The order of both lists align with each other.
    """

    polygons: List[List[data.Polygon]]
    classes: List[List[str]]
    model_info: data.ModelInfo


########## Pose Estimation ##########
class PoseEstimationRequest(BaseModel):
    """Pose estimation RPC request."""

    base64_imgs: List[str]  # base64 encoded images


class PoseEstimationResponse(BaseModel):
    """Pose estimation RPC response.

    The order of both lists align with each other.
    """

    keypoints: List[List[data.HumanBodyKeypoints]]
    classes: List[List[str]]
    model_info: data.ModelInfo
