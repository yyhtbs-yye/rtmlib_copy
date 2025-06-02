#!/usr/bin/env python3
import asyncio
import logging
import json
import os
import time
import argparse

import av
import paho.mqtt.client as mqtt
from rtmlib import Body

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time pose estimation publisher from RTSP stream (async processing)."
    )
    parser.add_argument(
        '--rtsp_url',
        default=os.getenv('RTSP_URL', 'rtsp://192.168.200.206:8554/mystream'),
        help='RTSP URL'
    )
    parser.add_argument(
        '--broker',
        default=os.getenv('BROKER_URL', 'mqtt://192.168.200.206:1883'),
        help='MQTT broker URL (mqtt://host:port)'
    )
    parser.add_argument(
        '--topic',
        default=os.getenv('TOPIC', 'rtmpose'),
        help='MQTT topic to publish pose data'
    )
    parser.add_argument(
        '--backend',
        default=os.getenv('BACKEND', 'onnxruntime'),
        help='Inference backend for rtmlib.Body (opencv, onnxruntime, openvino)'
    )
    parser.add_argument(
        '--queue_size',
        type=int,
        default=int(os.getenv('QUEUE_SIZE', '8')),
        help='Async frame queue max size'
    )
    parser.add_argument(
        '--cuda_visible_devices',
        default=os.getenv('CUDA_VISIBLE_DEVICES', '0,1,2,3'),
        help='Comma-separated list of CUDA devices to use (e.g. 0,1,2,3)'
    )
    return parser.parse_args()

async def worker_loop(worker_id: int,
                      body: Body,
                      frame_q: asyncio.Queue,
                      client: mqtt.Client,
                      topic: str):
    """
    Pull (frame_id, frame_np) from frame_q, run async pose inference,
    and publish JSON payload via MQTT.
    """
    loop = asyncio.get_running_loop()
    while True:
        item = await frame_q.get()
        if item is None:
            break

        frame_id, frame_np = item

        height, width = frame_np.shape[:2]

        scale = 1.0

        tic = time.time()
        # Run Body() on this worker’s dedicated GPU in executor
        bboxes, bbox_scores, keypoints, keypoint_scores = await loop.run_in_executor(
            None, lambda: body(frame_np)
        )
        fps = 1.0 / (time.time() - tic) if tic else 0.0

        # Convert keypoints and scores to nested lists for JSON serialization
        # keypoints: list of (N, K, 2) arrays or similar; ensure each is python list
        bboxes_list = [bx.tolist() for bx in bboxes]
        keypoints_list = [kp.tolist() for kp in keypoints]
        bbox_scores_list = [bsc.tolist() for bsc in bbox_scores]
        keypoint_scores_list = [ksc.tolist() for ksc in keypoint_scores]

        client.publish(
            f"{topic}/bboxes",
            json.dumps({
                'frame_id': frame_id,
                'single_gpu_fps': fps,
                'scale': scale,
                'bboxes': bboxes_list,
                'bboxe_scores': bbox_scores_list
            }),
            qos=0
        )

        # Publish to 'keypoints' subtopic
        client.publish(
            f"{topic}/keypoints",
            json.dumps({
                'frame_id': frame_id,
                'single_gpu_fps': fps,
                'scale': scale,
                'keypoints': keypoints_list,
                'keypoint_scores': keypoint_scores_list
            }),
            qos=0
        )
async def main():
    args = parse_args()

    # Parse MQTT broker URL
    proto, rest = args.broker.split('://', 1)
    host, port = rest.split(':')
    mqtt_client = mqtt.Client()
    mqtt_client.connect(host, int(port))
    mqtt_client.loop_start()

    # Configure GPU devices for workers
    GPU_DEVICES = [f"cuda:{i}" for i in range(len(os.getenv('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')))]
    NUM_WORKERS = len(GPU_DEVICES)

    # Instantiate one Body() per GPU
    bodies = [
        Body(
            det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            det_input_size=(640, 640),
            pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                 "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
            pose_input_size=(192, 256),
            mode="lightweight",
            backend=args.backend,
            device=gpu_id
        )
        for gpu_id in GPU_DEVICES
    ]

    frame_q = asyncio.Queue(maxsize=args.queue_size)

    # Launch async worker tasks
    worker_tasks = [
        asyncio.create_task(worker_loop(i, bodies[i], frame_q, mqtt_client, args.topic))
        for i in range(NUM_WORKERS)
    ]

    while True:
        try:
            container = av.open(args.rtsp_url)
            video_stream = next(s for s in container.streams if s.type == 'video')

            for packet in container.demux(video_stream):
                for frame in packet.decode():
                    # Extract SEI_UNREGISTERED for frame_id if present
                    frame_id = None
                    if frame.side_data:
                        for side_data in frame.side_data:
                            if side_data.type.name == "SEI_UNREGISTERED":
                                frame_id = bytes(side_data).decode('ascii', 'ignore')

                    # Convert AV frame to BGR numpy array
                    frame_np = frame.to_ndarray(format='bgr24')

                    # Put into async queue (await if full)
                    await frame_q.put((frame_id, frame_np))

            # When stream ends, send poison pills to workers
            for _ in range(NUM_WORKERS):
                await frame_q.put(None)

            # Wait for all workers to finish before trying to reconnect
            await asyncio.gather(*worker_tasks)
            container.close()
            break  # Exit if the stream ended normally

        except Exception:
            logging.exception("Error processing RTSP stream; reconnecting in 5s")
            await asyncio.sleep(5)

    mqtt_client.loop_stop()
    mqtt_client.disconnect()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
