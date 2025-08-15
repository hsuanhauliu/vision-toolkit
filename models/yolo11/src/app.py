# Run this script from the parent directory
# python ./src/inference_img.py


# NOTE: Assuming ObjectDetectionRequest exists in rpc.py, similar to ObjectDetectionResponse
from rpc import ObjectDetectionResponse, ObjectDetectionRequest
from inference import get_inference_func, get_response_model

import os
import logging

from fast_serve import create_app
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

LOGGER = logging.getLogger(__name__)
# Configure logging to show INFO level messages for better debugging
logging.basicConfig(level=logging.INFO)

YOLO_TASK = os.environ.get("YOLO_TASK", "instance_segmentation").lower()

inference_func = get_inference_func(YOLO_TASK)
response_model = get_response_model(YOLO_TASK)

LOGGER.info(f"Preparing API endpoints for {YOLO_TASK} task")
app = create_app(
    inference_func,
    response_model=response_model,
    http_endpoint="/predict",
    websocket_endpoint="/ws/predict",
)

# Add necessary middleware and static files to the app instance
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
