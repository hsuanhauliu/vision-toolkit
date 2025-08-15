# NOTE: This docker file needs to be run from the root directory of this repo.
#
# =====================================================================
# Stage 1: Final Runtime Stage
#
# Using a slim base image is a great way to keep the final image
# size small while maintaining good compatibility with Python packages.
# =====================================================================
FROM python:3.11-slim-bullseye

# --- Environment Variables ---
# These prevent Python from writing .pyc files and ensure output
# is sent straight to the terminal without buffering, which is
# useful for logging in containerized applications.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# --- Set Working Directory ---
# Sets the default directory for all subsequent commands.
WORKDIR /app

# --- Install Dependencies ---
RUN apt-get update && apt-get install git ffmpeg libsm6 libxext6  -y

# --no-cache-dir is used to reduce the image size by not storing the cache.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir git+https://github.com/hsuanhauliu/fast-serve.git@v0.2.0 && \
    pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir ultralytics && \
    pip install --no-cache-dir uvicorn[standard]

# --- Copy Application Code ---
# copy backend source code
COPY ./models/yolo11/src .

# Frontend client
ARG CLIENT="image_classification"

RUN mkdir -p ./static

COPY ./docs/${CLIENT}/index.html ./static

# --- Entrypoint ---
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=5000"]
