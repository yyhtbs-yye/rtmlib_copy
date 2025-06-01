#!/usr/bin/env python3
import asyncio
import cv2
import time
import os
from rtmlib import Body
from tqdm import tqdm

VIDEO_PATH   = "rte_far_seg_1.mp4"
BACKEND      = "onnxruntime"         # opencv, onnxruntime, openvino
GPU_DEVICES  = [f"cuda:{i}" for i in range(4)]   # cuda:0 … cuda:3
NUM_WORKERS  = len(GPU_DEVICES)      # 4 async workers
QUEUE_SIZE   = 8                     # tune for throughput

os.makedirs("output_frames", exist_ok=True)

# Build one pose-estimator per GPU
bodies = [
    Body(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        det_input_size=(640, 640),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
             "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        pose_input_size=(192, 256),
        mode="lightweight",            # balanced / performance / lightweight
        backend=BACKEND,
        device=gpu_id,
    )
    for gpu_id in GPU_DEVICES
]

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


async def producer(frame_q: asyncio.Queue):
    """
    Read frames synchronously (blocking) and push them into the queue.
    This keeps the main loop line-by-line as normal execution.
    """
    idx = 0
    while True:
        ok, frame = cap.read()           # synchronous, blocking
        if not ok:
            break
        idx += 1
        await frame_q.put((idx, frame))

    # Send poison pills so workers know when to stop
    for _ in range(NUM_WORKERS):
        await frame_q.put(None)


async def worker(worker_id: int,
                 body: Body,
                 frame_q: asyncio.Queue,
                 result_q: asyncio.Queue):
    """
    Run pose inference for frames pulled from frame_q.
    """
    loop = asyncio.get_running_loop()
    while True:
        item = await frame_q.get()
        if item is None:
            # Signal the printer that this worker is done
            await result_q.put(None)
            break

        idx, frame = item
        tic = time.time()
        # run Body() on this worker’s dedicated GPU asynchronously
        keypoints, scores = await loop.run_in_executor(
            None, lambda: body(frame)
        )
        fps = 1.0 / (time.time() - tic) if tic else 0.0
        await result_q.put((idx, fps))


async def printer(result_q: asyncio.Queue):
    """
    Display per-frame FPS in order, with a tqdm progress bar.
    """
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    next_expected, buffer, done = 1, {}, 0

    while done < NUM_WORKERS:
        item = await result_q.get()
        if item is None:
            done += 1
            continue

        idx, fps = item
        if idx == next_expected:
            tqdm.write(f"Frame {idx}: {fps:.2f} FPS")
            pbar.update(1)
            next_expected += 1
            # flush any buffered out-of-order items
            while next_expected in buffer:
                tqdm.write(f"Frame {next_expected}: {buffer.pop(next_expected):.2f} FPS")
                pbar.update(1)
                next_expected += 1
        else:
            buffer[idx] = fps

    pbar.close()


async def main():
    frame_q  = asyncio.Queue(maxsize=QUEUE_SIZE)
    result_q = asyncio.Queue()

    # Launch producer + workers + printer
    tasks = [
        asyncio.create_task(producer(frame_q)),
        *[
            asyncio.create_task(worker(i, bodies[i], frame_q, result_q))
            for i in range(NUM_WORKERS)
        ],
        asyncio.create_task(printer(result_q)),
    ]

    await asyncio.gather(*tasks)
    cap.release()


if __name__ == "__main__":
    asyncio.run(main())
