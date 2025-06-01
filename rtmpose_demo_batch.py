import time
import cv2
from rtmlib.tools.solution.body_batch import Body
from rtmlib import draw_skeleton

from tqdm import tqdm
import os

os.makedirs('output_frames', exist_ok=True)

device = 'cuda:0'  # 'cuda:0' for GPU, 'cpu' for CPU
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

K = 5  # or any buffer size you want
frame_buffer = []

with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_buffer.append(frame)
        frame_idx += 1
        pbar.update(1)

        if len(frame_buffer) == K:
            height, width = frame.shape[:2]

            start_time = time.time()
            keypoints_list, scores_list = body(frame_buffer)  # Assume body() accepts a list of frames now
            elapsed_time = time.time() - start_time

            tqdm.write(f"Frames {frame_idx - K + 1}-{frame_idx}: Latency {elapsed_time:.2f}s")

            # for j, (keypoints, scores) in enumerate(zip(keypoints_list, scores_list)):
            #     frame = frame_buffer[j]
            #     # Draw the skeleton on the image
            #     img_show = draw_skeleton(
            #         frame,
            #         keypoints,
            #         scores,
            #         openpose_skeleton=openpose_skeleton,
            #         kpt_thr=0.1,
            #         line_width=2
            #     )

            #     # Optionally resize back to original dimensions (usually not needed here)
            #     img_show = cv2.resize(img_show, (width, height))

            #     # Save the result
            #     output_path = f'output_frames/rte_close_seg_1/frame_{frame_idx - K + j:08d}.png'
            #     cv2.imwrite(output_path, img_show)
            
            frame_buffer = []  # Clear buffer


    # Optionally process remaining frames in buffer
    if frame_buffer:
        start_time = time.time()
        keypoints_list, scores_list = body(frame_buffer)
        elapsed_time = time.time() - start_time

        tqdm.write(f"Remaining {len(frame_buffer)} Latency: {elapsed_time:.2f}s")

cap.release()
