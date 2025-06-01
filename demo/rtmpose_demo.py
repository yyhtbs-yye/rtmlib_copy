import time
import cv2
from rtmlib import Body
from tqdm import tqdm
import os

os.makedirs('output_frames', exist_ok=True)

device = 'cuda'  # 'cuda:0' for GPU, 'cpu' for CPU
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

video_path = "rte_far_seg_1.mp4"
cap = cv2.VideoCapture(video_path)

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

body = Body(det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
            det_input_size=(640, 640),
            pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
            pose_input_size=(192, 256),
            mode='lightweight',  # balanced, performance, lightweight
            backend=backend,
            device=device)

frame_idx = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        height, width = frame.shape[:2]

        start_time = time.time()
        keypoints, scores = body(frame)
        elapsed_time = time.time() - start_time

        fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        tqdm.write(f"Frame {frame_idx}: {fps:.2f} FPS")

        pbar.update(1)

cap.release()
