import pynvjpeg
import torch
import cv2
import queue
from threading import Thread
from tqdm import tqdm
import time
import numpy as np


class _EncoderThread(Thread):
    """
    A dedicated worker thread that owns a single, complete nvJPEG instance.
    It runs a simple, synchronous loop, processing frames from a shared input queue.
    """

    def __init__(self, input_queue, output_queue, quality, subsampling, device_id=0):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.quality = quality
        self.subsampling = subsampling
        self.device_id = device_id
        self.is_running = True

        # Each thread initializes its own resources upon starting.
        self.handle = None
        self.state = None
        self.params = None
        self.stream_ptr = torch.cuda.Stream().cuda_stream

    def run(self):
        """The main loop for this worker thread."""
        torch.cuda.set_device(self.device_id)

        # --- Per-Thread Initialization ---
        status, self.handle = pynvjpeg.nvjpeg_create_simple()
        status, self.state = pynvjpeg.nvjpeg_encoder_state_create(
            self.handle, self.stream_ptr
        )
        status, self.params = pynvjpeg.nvjpeg_encoder_params_create(
            self.handle, self.stream_ptr
        )
        pynvjpeg.nvjpeg_encoder_params_set_quality(
            self.params, self.quality, self.stream_ptr
        )
        pynvjpeg.nvjpeg_encoder_params_set_sampling_factors(
            self.params, self.subsampling, self.stream_ptr
        )

        while self.is_running:
            try:
                # Block until a job is available
                image_tensor = self.input_queue.get(timeout=0.1)

                # --- Synchronous Encode/Decode Cycle ---
                height, width, _ = image_tensor.shape

                nv_image = pynvjpeg.NvjpegImage()
                pynvjpeg.nvjpeg_image_set_channel(
                    nv_image, 0, image_tensor.data_ptr(), width * 3
                )

                pynvjpeg.nvjpeg_encode_image(
                    self.handle,
                    self.state,
                    self.params,
                    nv_image,
                    pynvjpeg.nvjpegInputFormat_t.RGBI,
                    width,
                    height,
                    self.stream_ptr,
                )

                # This is a blocking call, but it only blocks this thread.
                status, actual_size = (
                    pynvjpeg.nvjpeg_encode_retrieve_bitstream_with_size(
                        self.handle, self.state, 0, 0, self.stream_ptr
                    )
                )

                if status == pynvjpeg.nvjpegStatus_t.SUCCESS and actual_size > 0:
                    jpeg_data = torch.empty(
                        actual_size, dtype=torch.uint8, device="cpu"
                    )
                    pynvjpeg.nvjpeg_encode_retrieve_bitstream_with_size(
                        self.handle,
                        self.state,
                        jpeg_data.data_ptr(),
                        actual_size,
                        self.stream_ptr,
                    )
                    self.output_queue.put(jpeg_data)

                self.input_queue.task_done()

            except queue.Empty:
                # If the queue is empty, check if we should shut down.
                if not self.is_running and self.input_queue.empty():
                    break

        self.shutdown()

    def stop(self):
        self.is_running = False

    def shutdown(self):
        if self.handle:
            pynvjpeg.nvjpeg_encoder_params_destroy(self.params)
            pynvjpeg.nvjpeg_encoder_state_destroy(self.state)
            pynvjpeg.nvjpeg_destroy(self.handle)
            self.handle = None


class NvjpegService:
    """Manages a pool of dedicated encoder threads."""

    def __init__(
        self,
        quality: int,
        subsampling: pynvjpeg.nvjpegChromaSubsampling_t,
        num_threads: int = 4,
    ):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        self.workers = []
        for _ in range(num_threads):
            worker = _EncoderThread(
                self.input_queue, self.output_queue, quality, subsampling
            )
            self.workers.append(worker)

    def start(self):
        for worker in self.workers:
            worker.start()

    def submit(self, image_tensor):
        self.input_queue.put(image_tensor)

    def get_result(self, block=True, timeout=None):
        return self.output_queue.get(block=block, timeout=timeout)

    def shutdown(self):
        print("Shutting down service...")
        # Signal all workers to stop processing new items
        for worker in self.workers:
            worker.stop()

        # Wait for all threads to finish their current tasks and exit
        for worker in self.workers:
            worker.join()
        print("All workers shut down.")


if __name__ == "__main__":
    # Create a dummy video frame
    cap = cv2.VideoCapture("test.mp4")
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1080, 1920))
    image_tensor = torch.from_numpy(img).cuda()

    # --- Initialize the Service ---
    jpeg_service = NvjpegService(
        quality=50,
        subsampling=pynvjpeg.nvjpegChromaSubsampling_t.CSS_420,
        num_threads=4,
    )

    # --- Submit Jobs ---
    num_frames_to_encode = 500000
    print(f"Submitting {num_frames_to_encode} frames for encoding...")
    for _ in tqdm(range(num_frames_to_encode)):
        jpeg_service.input_queue.put(image_tensor)

    jpeg_service.start()
    # --- Retrieve Results ---
    print("Retrieving encoded frames...")
    encoded_frames = []
    with tqdm(total=num_frames_to_encode, smoothing=0.02) as pbar:
        for _ in range(num_frames_to_encode):
            jpeg_data = (
                jpeg_service.output_queue.get()
            )  # .get() will block until an item is available
            # encoded_frames.append(jpeg_data)
            pbar.update(1)

    # --- Shutdown ---
    print("Stopping service...")
    jpeg_service.stop()
    jpeg_service.join()  # Wait for the thread to finish processing all items

    print("\nEncoding complete!")
    print(f"Processed {len(encoded_frames)} frames.")
    if encoded_frames:
        # Save the last frame for verification
        with open("final_frame.jpg", "wb") as f:
            f.write(encoded_frames[-1].numpy())
        print(
            f"Size of last encoded frame: {len(encoded_frames[-1])} bytes. Saved to 'final_frame.jpg'."
        )
