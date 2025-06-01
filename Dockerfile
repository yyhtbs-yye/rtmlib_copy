# 1. Base image with Python 3.11
FROM continuumio/miniconda3:latest


# 2. Set up environment variables with sane defaults for the new script
ENV RTSP_URL="rtsp://192.168.200.206:8554/mystream" \
    BROKER_URL="mqtt://192.168.200.206:1883" \
    TOPIC="poses/frames" \
    BACKEND="onnxruntime" \
    QUEUE_SIZE="8" \
    CUDA_VISIBLE_DEVICES="0,1,2,3"

# 3. Install system-level dependencies (PyAV, OpenCV, FFmpeg, etc.)

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
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

# 4. Set working directory
WORKDIR /app

RUN conda init

RUN pip install --no-cache-dir \
        av==14.4.0 \
        numpy==1.26.2 \
        onnxruntime==1.22.0 \
        opencv-contrib-python==4.11.0.86 \
        opencv-python==4.11.0.86 \
        paho-mqtt==2.1.0 \
        tqdm

# 5. Clone and install rtmlib (the Body-based pose estimator)
RUN git clone https://github.com/yyhtbs-yye/rtmlib_copy

RUN cd rtmlib_copy && \
    pip install --no-cache-dir -e .


RUN pip install --no-cache-dir \
        onnxruntime-gpu==1.22.0 

# RUN pip uninstall -y onnxruntime-gpu

# RUN pip install --no-cache-dir \
#         onnxruntime-gpu==1.22.0 

RUN conda install -c conda-forge cudnn=9.0.0 

# 7. Copy the RTSP→RTMPose→MQTT publisher script into the image
COPY rtsp_rtmpose_gpus_pub.py .

# 8. Default command to run the publisher with environment-driven args
CMD ["sh", "-c", "\
    python rtsp_rtmpose_gpus_pub.py \
      --rtsp_url \"${RTSP_URL}\" \
      --broker \"${BROKER_URL}\" \
      --topic \"${TOPIC}\" \
      --backend \"${BACKEND}\" \
      --queue_size \"${QUEUE_SIZE}\" \
      --cuda_visible_devices \"${CUDA_VISIBLE_DEVICES}\" \
"]
