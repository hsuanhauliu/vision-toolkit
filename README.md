# Vision Toolkit

Tools for Computer Vision experiments.

## YOLO11

Inference service serving Ultralytics' [YOLO11](https://docs.ultralytics.com/models/yolo11/) model built with Docker.

You can check out the frontend clients via the links below. Note that this is just the frontend app and not connected to any backend.

- https://hsuanhauliu.github.io/vision-toolkit/object_detection
- https://hsuanhauliu.github.io/vision-toolkit/image_classification
- https://hsuanhauliu.github.io/vision-toolkit/instance_segmentation
- https://hsuanhauliu.github.io/vision-toolkit/pose_estimation

## Run

Place the saved model in ./data folder. You can use the provided script to download YOLO11. Modify the URL in the script if you want to use different yolo models.

You can also go to their [release page](https://github.com/ultralytics/assets/releases) on Github and download model from there directly.

```bash
./get_yolo_model.sh # this will download the file in ./data directory
```

Build Docker image.

```bash
# Build the image. Needs to be in the current directory.
docker build -t yolo11 -f models/yolo11/Dockerfile .

# You can override the task by passing in BUILD_DIR_NAME. Default is object detection.
docker build --build-arg YOLO_TASK="object_detection" -t yolo11 -f models/yolo11/Dockerfile .
```

Run Docker container.

```bash
# Frontend client will be running on http://localhost:8000. The default will build object detection docker image.
# Note: right now the clients are hardcoded to use port 8000. You can modify the index.html to change that.
docker run --rm -v ./data:/app/data --name yolo11 -p 8000:5000 yolo11

# You can override the model task and saved model file name using environment variable like so:
# Note: the backend will search for saved model file in ./data directory. Default model name is yolo_model.pt
docker run --rm -v ./data:/app/data --name yolo11 -p 8000:5000 -e YOLO_TASK=object_detect -e SAVED_MODEL=yolo_model.pt yolo11
```
