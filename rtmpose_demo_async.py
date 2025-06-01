import asyncio
import cv2
import time
import os
from rtmlib import Body
from tqdm import tqdm

os.makedirs('output_frames', exist_ok=True)

device = 'cuda'  # 'cuda' for GPU, 'cpu' for CPU
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

video_path = "rte_far_seg_1.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize the pose-estimator once, outside of any async tasks
body = Body(
    det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
    det_input_size=(640, 640),
    pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
    pose_input_size=(192, 256),
    mode='lightweight',  # balanced, performance, lightweight
    backend=backend,
    device=device
)

async def producer(frame_queue: asyncio.Queue):
    """
    Read frames from `cap` in a thread so as not to block the event loop,
    and put (frame_idx, frame) into frame_queue. Once done, put N sentinels
    (None) for the workers to exit.
    """
    loop = asyncio.get_running_loop()
    idx = 0
    while True:
        # Read the next frame from cap in a threadpool
        success, frame = await loop.run_in_executor(None, cap.read)
        if not success:
            break
        idx += 1
        await frame_queue.put((idx, frame))
    # Signal “no more frames” by pushing None once per worker
    # (we’ll spawn `NUM_WORKERS` workers below)
    for _ in range(NUM_WORKERS):
        await frame_queue.put(None)

async def worker(frame_queue: asyncio.Queue, result_queue: asyncio.Queue):
    """
    Pull (idx, frame) from frame_queue, run `body(frame)` inside
    run_in_executor, compute FPS, and push (idx, fps) into result_queue.
    Exit when frame_queue gives None.
    """
    loop = asyncio.get_running_loop()
    while True:
        item = await frame_queue.get()
        if item is None:
            # Pass along the sentinel so printer knows how many results to expect
            await result_queue.put(None)
            break

        idx, frame = item
        start_time = time.time()

        # Call the synchronous body(frame) in a thread:
        keypoints, scores = await loop.run_in_executor(None, lambda: body(frame))

        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        await result_queue.put((idx, fps))

async def printer(result_queue: asyncio.Queue):
    """
    Pull (idx, fps) from result_queue, but hold onto any out-of-order items
    until we can emit them in exact ascending order. We know there will be
    exactly `NUM_WORKERS` None-sentinels at the end; once we've received all
    of them, we stop.
    """
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
    next_expected = 1
    buffer = {}
    dones = 0

    while dones < NUM_WORKERS:
        item = await result_queue.get()
        if item is None:
            dones += 1
            continue

        idx, fps = item
        if idx == next_expected:
            # We can print it right away
            tqdm.write(f"Frame {idx}: {fps:.2f} FPS")
            pbar.update(1)
            next_expected += 1

            # Check if the buffer has the next frames waiting
            while next_expected in buffer:
                buffered_fps = buffer.pop(next_expected)
                tqdm.write(f"Frame {next_expected}: {buffered_fps:.2f} FPS")
                pbar.update(1)
                next_expected += 1
        else:
            # Store out-of-order result
            buffer[idx] = fps

    pbar.close()

async def main():
    frame_queue = asyncio.Queue(maxsize=8)
    result_queue = asyncio.Queue()

    # Launch producer, NUM_WORKERS workers, and a single printer
    prod_task = asyncio.create_task(producer(frame_queue))
    worker_tasks = [asyncio.create_task(worker(frame_queue, result_queue))
                    for _ in range(NUM_WORKERS)]
    printer_task = asyncio.create_task(printer(result_queue))

    # Wait for everything to finish
    await prod_task
    await asyncio.gather(*worker_tasks)
    await printer_task

    cap.release()

if __name__ == "__main__":
    # You can tune this up or down depending on how many threads you want
    # working on body(frame) in parallel.
    NUM_WORKERS = 4
    asyncio.run(main())
