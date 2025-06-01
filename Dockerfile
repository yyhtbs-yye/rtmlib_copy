# 1. Base image with Python 3.10
FROM python:3.10-slim

# 2. Set up environment variables with sane defaults
ENV RTSP_URL="rtsp://192.168.200.206:8554/mystream" \
    MODEL_PATH="yolov8n.pt" \
    DEVICE="0" \
    RESIZE="640" \
    BROKER_URL="mqtt://192.168.200.206:1883" \
    TOPIC="objects/frames"

# 3. Install system-level deps for PyAV, OpenCV, FFmpeg, MQTT client, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ffmpeg \
      libsm6 \
      libxext6 \
      libglib2.0-0 \
      pkg-config \
      libavformat-dev \
      libavcodec-dev \
      libavutil-dev && \
    rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN cd 

# 5. Copy your publisher script
COPY rtsp_object_tracking_pub.py .

# 6. Expose nothing (this is a publisher-only container)
#    If using MQTT over TCP, you could expose 1883, but normally the broker is external.

# 7. Entrypoint: run your script, reading from env vars
CMD ["sh", "-c", "python rtsp_rtmpose_gpus_pub.py --url \"${RTSP_URL}\" --model \"${MODEL_PATH}\" --device \"${DEVICE}\" --resize \"${RESIZE}\" --broker \"${BROKER_URL}\" --topic \"${TOPIC}\""]
