#!/bin/bash

# Description: This script downloads the yolo11n.pt model from the official Ultralytics GitHub releases.

# --- Configuration ---
# The URL of the file to be downloaded.
FILE_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"

# The desired name for the downloaded file.
OUTPUT_FILENAME="./models/yolo11/data/yolo_model.pt"

# --- Script Body ---

echo "Checking for: $OUTPUT_FILENAME"

# --- Pre-download Check ---
# Check if the file already exists before attempting to download.
if [ -f "$OUTPUT_FILENAME" ]; then
    echo "✅ File '$OUTPUT_FILENAME' already exists. Skipping download."
    exit 0
fi

echo "Starting download for: $OUTPUT_FILENAME"
echo "From: $FILE_URL"
echo "--------------------------------------------------"

# Check if wget is installed. If it is, use it to download the file.
# The -c flag allows for resuming an interrupted download.
if command -v wget &> /dev/null
then
    echo "Attempting download with wget..."
    wget -c -O "$OUTPUT_FILENAME" "$FILE_URL"

    # Check the exit code of wget to see if it succeeded.
    if [ $? -eq 0 ]; then
        echo "--------------------------------------------------"
        echo "✅ Download successful!"
        echo "File saved as: $OUTPUT_FILENAME"
        exit 0
    else
        echo "⚠️ wget failed. Trying with curl as a fallback..."
    fi
fi

# If wget failed or isn't installed, try using curl.
# The -L flag tells curl to follow any redirects.
# The -o flag specifies the output filename.
if command -v curl &> /dev/null
then
    echo "Attempting download with curl..."
    curl -L -o "$OUTPUT_FILENAME" "$FILE_URL"

    # Check the exit code of curl.
    if [ $? -eq 0 ]; then
        echo "" # Add a newline for better formatting after curl's progress bar
        echo "--------------------------------------------------"
        echo "✅ Download successful!"
        echo "File saved as: $OUTPUT_FILENAME"
        exit 0
    else
        echo "" # Add a newline
        echo "❌ Error: curl command failed."
    fi
else
    echo "❌ Error: Neither wget nor curl are installed."
    echo "Please install one of them to proceed."
fi

# If both methods failed, exit with an error status.
echo "--------------------------------------------------"
echo "❌ Download failed. Please check the URL or your network connection."
exit 1
