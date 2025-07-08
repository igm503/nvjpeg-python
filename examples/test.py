import pynvjpeg
import torch
import cv2

print("Loading 'test.jpg'...")
img = cv2.imread("ian.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image_tensor = torch.from_numpy(img).to("cuda")

height, width, _ = image_tensor.shape

# initalize the encoder state vars + the stream we'll run on
stream = torch.cuda.Stream().cuda_stream
status, handle = pynvjpeg.nvjpeg_create_simple()
status, encoder_state = pynvjpeg.nvjpeg_encoder_state_create(handle, stream)
status, encoder_params = pynvjpeg.nvjpeg_encoder_params_create(handle, stream)

# Set parameters
pynvjpeg.nvjpeg_encoder_params_set_quality(encoder_params, 10, stream)
pynvjpeg.nvjpeg_encoder_params_set_sampling_factors(
    encoder_params, pynvjpeg.nvjpegChromaSubsampling_t.CSS_420, stream
)

# You can get the max buffer size needed for the output if you don't want to
# have to sync later. See below.
status, max_size = pynvjpeg.nvjpeg_encode_get_buffer_size(
    handle, encoder_params, width, height
)
print(f"Max buffer size: status={status}, max_size={max_size}")

if status == pynvjpeg.nvjpegStatus_t.SUCCESS:
    nv_image = pynvjpeg.NvjpegImage()
    # You have to set the image data. This is clunky, but for interleaved RGB, you
    # only set channel 0.
    # The stride is the number of bytes per row: width * num_channels
    pynvjpeg.nvjpeg_image_set_channel(nv_image, 0, image_tensor.data_ptr(), width * 3)

    print("Encoding...")
    status = pynvjpeg.nvjpeg_encode_image(
        handle,
        encoder_state,
        encoder_params,
        nv_image,
        pynvjpeg.nvjpegInputFormat_t.RGBI,  # Use RGBI, where 'I' stands for interleaved (I think!)
        width,
        height,
        stream,
    )
    print(f"Encode: {status}")

    if status == pynvjpeg.nvjpegStatus_t.SUCCESS:
        # need to synchronize stream to get size. We can skip this sync by instead using
        # the max size from nvjpeg_encode_get_buffer_size to create the buffer
        torch.cuda.synchronize()

        # Get the exact size of the compressed bitstream
        status, actual_size = pynvjpeg.nvjpeg_encode_retrieve_bitstream_size(
            handle, encoder_state, stream
        )
        print(f"Get actual size: status={status}, actual_size={actual_size}")

        if status == pynvjpeg.nvjpegStatus_t.SUCCESS and actual_size > 0:
            # Create a buffer to hold the compressed bitstream
            jpeg_data = torch.empty(actual_size, dtype=torch.uint8, device="cpu")

            # Retrieve the bitstream from the GPU
            pynvjpeg.nvjpeg_encode_retrieve_bitstream(
                handle,
                encoder_state,
                jpeg_data.data_ptr(),
                actual_size,
                stream,
            )
            torch.cuda.synchronize()

            print("Saving 'encoded.jpg'...")
            with open("encoded.jpg", "wb") as f:
                f.write(jpeg_data.cpu().numpy())
            print("Save complete.")

# make sure to destroy these...
pynvjpeg.nvjpeg_encoder_params_destroy(encoder_params)
pynvjpeg.nvjpeg_encoder_state_destroy(encoder_state)
pynvjpeg.nvjpeg_destroy(handle)
