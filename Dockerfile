# 1. Use NVIDIA CUDA 12.3 + cuDNN 9 runtime on Ubuntu 20.04
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# 2. Install system-level dependencies (git, build tools, FFmpeg, OpenCV prerequisites, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget \
      bzip2 \
      ca-certificates \
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

# 3. Install Miniconda3 into /opt/conda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    mkdir -p $CONDA_DIR && \
    bash /tmp/miniconda.sh -b -f -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# 4. Initialize conda so that "conda activate" works in later layers
SHELL ["/bin/bash", "-lc"]
RUN conda init bash

# 5. Set up environment variables for the RTSP→RTMPose→MQTT publisher
ENV RTSP_URL="rtsp://192.168.200.206:8554/mystream" \
    BROKER_URL="mqtt://192.168.200.206:1883" \
    TOPIC="poses/frames" \
    BACKEND="onnxruntime" \
    QUEUE_SIZE="8" \
    CUDA_VISIBLE_DEVICES="0,1,2,3"

# 6. (Optional) Create a dedicated conda environment called "poseenv"
RUN conda create -y -n poseenv python=3.10 && \
    conda clean -afy

SHELL ["/bin/bash", "-lc"]
RUN conda activate poseenv

# 7. Install Python dependencies via pip (onnxruntime-gpu will pick up the system’s CUDA 12 + cuDNN 9)
RUN conda activate poseenv && pip install --no-cache-dir \
        av==14.4.0 \
        numpy==1.26.2 \
        onnxruntime==1.22.0 \
        opencv-python==4.11.0.86 \
        opencv-contrib-python==4.11.0.86 \
        paho-mqtt==2.1.0 \
        tqdm

RUN conda activate poseenv && pip install onnxruntime-gpu==1.22.0

# 8. Clone and install rtmlib (Body-based pose estimator)
WORKDIR /app
RUN git clone https://github.com/yyhtbs-yye/rtmlib_copy && \
    cd rtmlib_copy && \
    conda activate poseenv && pip install -e .

# 9. Copy the RTSP→RTMPose→MQTT publisher script into the image
COPY rtsp_rtmpose_gpus_pub.py .

# 10. Default command to launch the publisher,
#     passing through environment-driven flags
CMD ["bash", "-lc", "\
    conda activate poseenv && \
    python rtsp_rtmpose_gpus_pub.py \
      --rtsp_url \"${RTSP_URL}\" \
      --broker \"${BROKER_URL}\" \
      --topic \"${TOPIC}\" \
      --backend \"${BACKEND}\" \
      --queue_size \"${QUEUE_SIZE}\" \
      --cuda_visible_devices \"${CUDA_VISIBLE_DEVICES}\" \
"]
