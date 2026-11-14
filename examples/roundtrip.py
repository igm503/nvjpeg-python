"""Decode a JPEG on the GPU with pynvjpeg, then re-encode it.

usage: python examples/roundtrip.py input.jpg [output.jpg] [quality]
"""

import sys

import numpy as np
import torch

import pynvjpeg as nj

S = nj.nvjpegStatus_t


def check(result):
    """Raise on a non-SUCCESS status; return the remaining outputs, if any."""
    status, *outputs = result if isinstance(result, tuple) else (result,)
    if status != S.SUCCESS:
        raise RuntimeError(f"nvJPEG call failed: {status}")
    return outputs[0] if len(outputs) == 1 else tuple(outputs) or None


def main(src, dst="roundtrip.jpg", quality=90):
    jpeg = open(src, "rb").read()
    stream = torch.cuda.Stream()
    s = stream.cuda_stream

    handle = check(nj.nvjpeg_create_simple())
    n, subsampling, widths, heights = check(nj.nvjpeg_get_image_info(handle, jpeg))
    w, h = widths[0], heights[0]
    print(f"{src}: {w}x{h}, {n} components, {subsampling}")

    # Decode to an interleaved RGB tensor on the GPU.
    state = check(nj.nvjpeg_jpeg_state_create(handle))
    rgb = torch.empty((h, w, 3), dtype=torch.uint8, device="cuda")
    image = nj.NvjpegImage([rgb.data_ptr()], [w * 3])
    check(nj.nvjpeg_decode(handle, state, jpeg, None, nj.nvjpegOutputFormat_t.RGBI, image, s))

    # Encode it again (same stream, so no sync needed in between).
    enc_state = check(nj.nvjpeg_encoder_state_create(handle, s))
    params = check(nj.nvjpeg_encoder_params_create(handle, s))
    check(nj.nvjpeg_encoder_params_set_quality(params, quality, s))
    check(nj.nvjpeg_encoder_params_set_sampling_factors(params, nj.nvjpegChromaSubsampling_t.CSS_420, s))
    check(nj.nvjpeg_encode_image(handle, enc_state, params, image, nj.nvjpegInputFormat_t.RGBI, w, h, s))

    size = check(nj.nvjpeg_encode_retrieve_bitstream(handle, enc_state, 0, 0, s))
    out = np.empty(size, np.uint8)
    size = check(nj.nvjpeg_encode_retrieve_bitstream(handle, enc_state, out.ctypes.data, size, s))
    stream.synchronize()
    out[:size].tofile(dst)
    print(f"wrote {dst}: {size} bytes (quality {quality})")

    check(nj.nvjpeg_encoder_params_destroy(params))
    check(nj.nvjpeg_encoder_state_destroy(enc_state))
    check(nj.nvjpeg_jpeg_state_destroy(state))
    check(nj.nvjpeg_destroy(handle))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], *(sys.argv[2:3]), *(int(q) for q in sys.argv[3:4]))
