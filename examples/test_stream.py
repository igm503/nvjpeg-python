import pynvjpeg
import torch
import cv2
import queue
from threading import Thread
from tqdm import tqdm
import time
import numpy as np


class NvjpegService(Thread):
    """
    An optimized, single-thread service for asynchronous JPEG encoding using
    buffer pooling and a combined retrieval call to maximize performance.
    """

    def __init__(
        self,
        quality: int,
        subsampling: pynvjpeg.nvjpegChromaSubsampling_t,
        num_streams: int = 4,
    ):
        super().__init__()
        self.is_running = True
        self.num_streams = num_streams

        status, self.handle = pynvjpeg.nvjpeg_create_simple()
        if status != pynvjpeg.nvjpegStatus_t.SUCCESS:
            raise RuntimeError(f"Failed to create nvJPEG handle: {status}")

        self.streams = []
        for _ in range(num_streams):
            stream_obj = torch.cuda.Stream()
            stream_ptr = stream_obj.cuda_stream
            status, state = pynvjpeg.nvjpeg_encoder_state_create(
                self.handle, stream_ptr
            )
            status, params = pynvjpeg.nvjpeg_encoder_params_create(
                self.handle, stream_ptr
            )
            pynvjpeg.nvjpeg_encoder_params_set_quality(params, quality, stream_ptr)
            pynvjpeg.nvjpeg_encoder_params_set_sampling_factors(
                params, subsampling, stream_ptr
            )
            self.streams.append(
                {
                    "stream_obj": stream_obj,
                    "stream_ptr": stream_ptr,
                    "state": state,
                    "params": params,
                }
            )

        self.inputs = queue.Queue()
        self.outputs = queue.Queue()
        self.buffer_pools = {}

    def _get_or_create_buffer_pool(self, width: int, height: int):
        """Creates a pool of reusable output buffers for a given resolution."""
        if (width, height) not in self.buffer_pools:
            print(f"Creating buffer pool for resolution {width}x{height}...")
            params = self.streams[0]["params"]
            status, max_size = pynvjpeg.nvjpeg_encode_get_buffer_size(
                self.handle, params, width, height
            )
            if status != pynvjpeg.nvjpegStatus_t.SUCCESS:
                raise RuntimeError(
                    f"Failed to get buffer size for {width}x{height}: {status}"
                )

            pool = queue.Queue()
            for _ in range(self.num_streams):
                buffer = torch.empty(
                    max_size, dtype=torch.uint8, device="cpu"
                ).pin_memory()
                pool.put(buffer)
            self.buffer_pools[(width, height)] = pool

        return self.buffer_pools[(width, height)]

    def run(self):
        """Main processing loop."""
        pending_jobs = {}

        while self.is_running or not self.inputs.empty() or pending_jobs:
            # --- Stage 1: Submit new encoding jobs ---
            for stream_idx in range(self.num_streams):
                if stream_idx in pending_jobs:
                    continue

                try:
                    image_tensor = self.inputs.get(timeout=0.001)
                    height, width, _ = image_tensor.shape

                    stream_obj = self.streams[stream_idx]["stream_obj"]
                    with torch.cuda.stream(stream_obj):
                        gpu_tensor = image_tensor.to("cuda", non_blocking=True)

                    stream_info = self.streams[stream_idx]
                    buffer_pool = self._get_or_create_buffer_pool(width, height)
                    output_buffer = buffer_pool.get()

                    nv_image = pynvjpeg.NvjpegImage()
                    pynvjpeg.nvjpeg_image_set_channel(
                        nv_image, 0, gpu_tensor.data_ptr(), width * 3
                    )

                    pynvjpeg.nvjpeg_encode_image(
                        self.handle,
                        stream_info["state"],
                        stream_info["params"],
                        nv_image,
                        pynvjpeg.nvjpegInputFormat_t.RGBI,
                        width,
                        height,
                        stream_info["stream_ptr"],
                    )

                    event = torch.cuda.Event()
                    event.record(stream_obj)

                    pending_jobs[stream_idx] = (event, output_buffer, buffer_pool)

                except queue.Empty:
                    break

            # --- Stage 2: Check for completed jobs and retrieve results ---
            for stream_idx in list(pending_jobs.keys()):
                event, output_buffer, pool = pending_jobs[stream_idx]

                if event.query():
                    stream_info = self.streams[stream_idx]

                    status, actual_size = (
                        pynvjpeg.nvjpeg_encode_retrieve_bitstream_with_size(
                            self.handle,
                            stream_info["state"],
                            output_buffer.data_ptr(),
                            output_buffer.numel(),
                            stream_info["stream_ptr"],
                        )
                    )

                    if status == pynvjpeg.nvjpegStatus_t.SUCCESS:
                        self.outputs.put(output_buffer[:100])
                        self.outputs.put(output_buffer[:actual_size])

                    pool.put(output_buffer)
                    del pending_jobs[stream_idx]

            time.sleep(0)

    def stop(self):
        self.is_running = False

    def shutdown(self):
        print("Shutting down NvjpegService...")
        for stream_info in self.streams:
            pynvjpeg.nvjpeg_encoder_params_destroy(stream_info["params"])
            pynvjpeg.nvjpeg_encoder_state_destroy(stream_info["state"])
        pynvjpeg.nvjpeg_destroy(self.handle)

    def __del__(self):
        if hasattr(self, "handle"):
            self.shutdown()


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
        num_streams=8,
    )

    # --- Submit Jobs ---
    num_frames_to_encode = 500000
    print(f"Submitting {num_frames_to_encode} frames for encoding...")
    for _ in tqdm(range(num_frames_to_encode)):
        jpeg_service.inputs.put(image_tensor)

    jpeg_service.start()
    # --- Retrieve Results ---
    print("Retrieving encoded frames...")
    encoded_frames = []
    with tqdm(total=num_frames_to_encode, smoothing=0.02) as pbar:
        for _ in range(num_frames_to_encode):
            jpeg_data = (
                jpeg_service.outputs.get()
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
