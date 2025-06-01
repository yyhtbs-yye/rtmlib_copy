docker run -d --gpus '"device=2,3,4,5"' \
  -e RTSP_URL="rtsp://192.168.200.206:8554/mystream" \
  -e BROKER_URL="mqtt://192.168.200.206:1883" \
  -e TOPIC="cam1/pose" \
  -e QUEUE_SIZE="10" \
  -e BACKEND="onnxruntime" \
  -e CUDA_VISIBLE_DEVICES="0,1,2,3" \
  rtsp_rtmpose_gpus_pub