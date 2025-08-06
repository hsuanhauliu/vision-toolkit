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


def keypointsToHumanBodyKeypoints(keypoints):
    keypoints = keypoints.xy[0]
    keypoint_coordinates = [data.Coordinate(x=p[0], y=p[1]) for p in keypoints]

    return data.HumanBodyKeypoints(
        nose=keypoint_coordinates[0],
        left_eye=keypoint_coordinates[1],
        right_eye=keypoint_coordinates[2],
        left_ear=keypoint_coordinates[3],
        right_ear=keypoint_coordinates[4],
        left_shoulder=keypoint_coordinates[5],
        right_shoulder=keypoint_coordinates[6],
        left_elbow=keypoint_coordinates[7],
        right_elbow=keypoint_coordinates[8],
        left_wrist=keypoint_coordinates[9],
        right_wrist=keypoint_coordinates[10],
        left_hip=keypoint_coordinates[11],
        right_hip=keypoint_coordinates[12],
        left_knee=keypoint_coordinates[13],
        right_knee=keypoint_coordinates[14],
        left_ankle=keypoint_coordinates[15],
        right_ankle=keypoint_coordinates[16],
    )


def detection_results_to_lists(model_results, class_names):
    """
    Convert the model inference results to a list of bounding boxes and a list of corresponding classes.
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


def classification_results_to_lists(model_results, class_names):
    """
    Convert the model inference results to a list of classes.
    """
    all_classes, all_probs = [], []
    for result in model_results:
        classes, probs = [], []
        for cls in result.probs.top5:
            classes.append(class_names[cls])
            probs.append(result.probs.data[cls])
        all_classes.append(classes)
        all_probs.append(probs)
    return all_classes, all_probs


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
