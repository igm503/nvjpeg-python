"""Encoding: all input formats, params, and bitstream retrieval paths."""

import cv2
import numpy as np
import pytest
import torch

import pynvjpeg as nj
from conftest import (
    S,
    cv_decode_rgb,
    cv_encode,
    interleaved_image,
    ok,
    planar_image,
    psnr,
)

CSS = nj.nvjpegChromaSubsampling_t
IF = nj.nvjpegInputFormat_t
ENC = nj.nvjpegJpegEncoding_t


@pytest.fixture
def encoder(handle, stream):
    s = stream.cuda_stream
    status, state = nj.nvjpeg_encoder_state_create(handle, s)
    ok(status)
    status, params = nj.nvjpeg_encoder_params_create(handle, s)
    ok(status)
    ok(nj.nvjpeg_encoder_params_set_quality(params, 95, s))
    ok(nj.nvjpeg_encoder_params_set_sampling_factors(params, CSS.CSS_444, s))
    yield state, params
    ok(nj.nvjpeg_encoder_params_destroy(params))
    ok(nj.nvjpeg_encoder_state_destroy(state))
    assert not params and not state


def retrieve(handle, state, stream):
    """Size query, then copy into an exactly-sized host buffer."""
    s = stream.cuda_stream
    status, size = nj.nvjpeg_encode_retrieve_bitstream(handle, state, 0, 0, s)
    ok(status)
    assert size > 0
    buf = np.empty(size, np.uint8)
    status, written = nj.nvjpeg_encode_retrieve_bitstream(handle, state, buf.ctypes.data, size, s)
    ok(status)
    stream.synchronize()
    assert written == size
    return buf.tobytes()


def test_encode_rgbi_roundtrip(handle, encoder, rgb, stream):
    state, params = encoder
    h, w, _ = rgb.shape
    src = torch.from_numpy(rgb).cuda()
    torch.cuda.synchronize()
    ok(nj.nvjpeg_encode_image(handle, state, params, interleaved_image(src), IF.RGBI, w, h, stream.cuda_stream))
    jpeg = retrieve(handle, state, stream)
    assert jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9"
    assert cv_decode_rgb(jpeg).shape == rgb.shape
    assert psnr(cv_decode_rgb(jpeg), rgb) > 35
    status, n, sub, widths, heights = nj.nvjpeg_get_image_info(handle, jpeg)
    ok(status)
    assert (n, sub, widths[0], heights[0]) == (3, CSS.CSS_444, w, h)


@pytest.mark.parametrize("fmt", [IF.RGB, IF.BGR, IF.RGBI, IF.BGRI])
def test_encode_input_formats(handle, encoder, rgb, stream, fmt):
    state, params = encoder
    h, w, _ = rgb.shape
    src = rgb[..., ::-1] if fmt in (IF.BGR, IF.BGRI) else rgb
    t = torch.from_numpy(np.ascontiguousarray(src)).cuda()
    if fmt in (IF.RGB, IF.BGR):
        t = t.permute(2, 0, 1).contiguous()
        img = planar_image(t)
    else:
        img = interleaved_image(t)
    torch.cuda.synchronize()
    ok(nj.nvjpeg_encode_image(handle, state, params, img, fmt, w, h, stream.cuda_stream))
    assert psnr(cv_decode_rgb(retrieve(handle, state, stream)), rgb) > 35


@pytest.mark.parametrize(
    "css,sampling", [(CSS.CSS_444, "444"), (CSS.CSS_422, "422"), (CSS.CSS_420, "420")]
)
def test_encode_sampling_factors(handle, encoder, rgb, stream, css, sampling):
    state, params = encoder
    s = stream.cuda_stream
    ok(nj.nvjpeg_encoder_params_set_sampling_factors(params, css, s))
    h, w, _ = rgb.shape
    t = torch.from_numpy(rgb).cuda()
    torch.cuda.synchronize()
    ok(nj.nvjpeg_encode_image(handle, state, params, interleaved_image(t), IF.RGBI, w, h, s))
    jpeg = retrieve(handle, state, stream)
    status, _, got_css, *_ = nj.nvjpeg_get_image_info(handle, jpeg)
    ok(status)
    assert got_css == css
    assert psnr(cv_decode_rgb(jpeg), rgb) > 33


def test_quality_changes_size(handle, encoder, rgb, stream):
    state, params = encoder
    s = stream.cuda_stream
    h, w, _ = rgb.shape
    t = torch.from_numpy(rgb).cuda()
    torch.cuda.synchronize()
    sizes = {}
    for q in (10, 50, 95):
        ok(nj.nvjpeg_encoder_params_set_quality(params, q, s))
        ok(nj.nvjpeg_encode_image(handle, state, params, interleaved_image(t), IF.RGBI, w, h, s))
        sizes[q] = len(retrieve(handle, state, stream))
    assert sizes[10] < sizes[50] < sizes[95]


def test_encoding_options(handle, encoder, rgb, stream):
    state, params = encoder
    s = stream.cuda_stream
    h, w, _ = rgb.shape
    t = torch.from_numpy(rgb).cuda()
    torch.cuda.synchronize()

    ok(nj.nvjpeg_encode_image(handle, state, params, interleaved_image(t), IF.RGBI, w, h, s))
    plain = retrieve(handle, state, stream)

    ok(nj.nvjpeg_encoder_params_set_optimized_huffman(params, 1, s))
    ok(nj.nvjpeg_encode_image(handle, state, params, interleaved_image(t), IF.RGBI, w, h, s))
    optimized = retrieve(handle, state, stream)
    assert len(optimized) < len(plain)
    assert psnr(cv_decode_rgb(optimized), cv_decode_rgb(plain)) == float("inf")

    # Accepted, but the nvJPEG 13 GPU encoder emits no DRI/RSTn markers
    # (verified with a plain C program), so only check the output is valid.
    ok(nj.nvjpeg_encoder_params_set_restart_interval(params, 8, s))
    ok(nj.nvjpeg_encode_image(handle, state, params, interleaved_image(t), IF.RGBI, w, h, s))
    with_rst = retrieve(handle, state, stream)
    assert psnr(cv_decode_rgb(with_rst), rgb) > 35
    ok(nj.nvjpeg_encoder_params_set_restart_interval(params, 0, s))

    ok(nj.nvjpeg_encoder_params_set_encoding(params, ENC.PROGRESSIVE_DCT_HUFFMAN, s))
    ok(nj.nvjpeg_encode_image(handle, state, params, interleaved_image(t), IF.RGBI, w, h, s))
    progressive = retrieve(handle, state, stream)
    status, js = nj.nvjpeg_jpeg_stream_create(handle)
    ok(status)
    ok(nj.nvjpeg_jpeg_stream_parse_header(handle, progressive, None, js))
    assert nj.nvjpeg_jpeg_stream_get_jpeg_encoding(js) == (S.SUCCESS, ENC.PROGRESSIVE_DCT_HUFFMAN)
    ok(nj.nvjpeg_jpeg_stream_destroy(js))
    assert psnr(cv_decode_rgb(progressive), rgb) > 35


def test_encode_yuv_planar(handle, encoder, rgb, stream):
    """nvjpegEncodeYUV from 4:2:0 planes produced by OpenCV's YCrCb conversion."""
    state, params = encoder
    s = stream.cuda_stream
    ok(nj.nvjpeg_encoder_params_set_sampling_factors(params, CSS.CSS_420, s))
    h, w, _ = rgb.shape
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y = torch.from_numpy(np.ascontiguousarray(ycrcb[..., 0])).cuda()
    cb = torch.from_numpy(cv2.resize(ycrcb[..., 2], (w // 2, h // 2), interpolation=cv2.INTER_AREA)).cuda()
    cr = torch.from_numpy(cv2.resize(ycrcb[..., 1], (w // 2, h // 2), interpolation=cv2.INTER_AREA)).cuda()
    torch.cuda.synchronize()
    img = nj.NvjpegImage([y.data_ptr(), cb.data_ptr(), cr.data_ptr()], [w, w // 2, w // 2])
    ok(nj.nvjpeg_encode_yuv(handle, state, params, img, CSS.CSS_420, w, h, s))
    jpeg = retrieve(handle, state, stream)
    assert psnr(cv_decode_rgb(jpeg), rgb) > 33


def test_encode_nv12(handle, encoder, rgb, stream):
    """nvjpegEncode with NV12 input (Y plane + interleaved UV plane)."""
    state, params = encoder
    s = stream.cuda_stream
    ok(nj.nvjpeg_encoder_params_set_sampling_factors(params, CSS.CSS_420, s))
    h, w, _ = rgb.shape
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cb = cv2.resize(ycrcb[..., 2], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    cr = cv2.resize(ycrcb[..., 1], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    uv = np.stack([cb, cr], axis=-1)
    y = torch.from_numpy(np.ascontiguousarray(ycrcb[..., 0])).cuda()
    uv_t = torch.from_numpy(np.ascontiguousarray(uv)).cuda()
    torch.cuda.synchronize()
    img = nj.NvjpegImage([y.data_ptr(), uv_t.data_ptr()], [w, w])
    status = nj.nvjpeg_encode(handle, state, params, img, CSS.CSS_420, IF.NV12, w, h, s)
    if status == S.IMPLEMENTATION_NOT_SUPPORTED:
        pytest.skip("NV12 encode not supported by this nvJPEG build")
    ok(status)
    assert psnr(cv_decode_rgb(retrieve(handle, state, stream)), rgb) > 33


def test_encode_rgbi_via_nvjpeg_encode(handle, encoder, rgb, stream):
    state, params = encoder
    h, w, _ = rgb.shape
    t = torch.from_numpy(rgb).cuda()
    torch.cuda.synchronize()
    status = nj.nvjpeg_encode(handle, state, params, interleaved_image(t), CSS.CSS_444, IF.RGBI, w, h, stream.cuda_stream)
    ok(status)
    assert psnr(cv_decode_rgb(retrieve(handle, state, stream)), rgb) > 35


def test_retrieve_paths_agree(handle, encoder, rgb, stream):
    """Host (exact / max-size / alias) and device retrieval give the same bytes.

    nvJPEG lets you copy out an encoded bitstream once (a second full retrieval
    returns garbage, verified in plain C), so re-encode before each path. Size
    queries (data=0) do not consume it.
    """
    state, params = encoder
    s = stream.cuda_stream
    h, w, _ = rgb.shape
    t = torch.from_numpy(rgb).cuda()
    torch.cuda.synchronize()

    def encode():
        ok(nj.nvjpeg_encode_image(handle, state, params, interleaved_image(t), IF.RGBI, w, h, s))

    encode()
    exact = retrieve(handle, state, stream)

    status, max_size = nj.nvjpeg_encode_get_buffer_size(handle, params, w, h)
    ok(status)
    assert max_size >= len(exact)
    pinned = torch.empty(max_size, dtype=torch.uint8).pin_memory()

    encode()
    status, n = nj.nvjpeg_encode_retrieve_bitstream(handle, state, pinned.data_ptr(), max_size, s)
    ok(status)
    stream.synchronize()
    assert pinned[:n].numpy().tobytes() == exact

    encode()
    assert nj.nvjpeg_encode_retrieve_bitstream_size(handle, state, s) == (S.SUCCESS, len(exact))
    pinned.zero_()
    status, n = nj.nvjpeg_encode_retrieve_bitstream_with_size(handle, state, pinned.data_ptr(), max_size, s)
    ok(status)
    stream.synchronize()
    assert pinned[:n].numpy().tobytes() == exact

    encode()
    status, dsize = nj.nvjpeg_encode_retrieve_bitstream_device(handle, state, 0, 0, s)
    ok(status)
    assert dsize == len(exact)
    dev = torch.empty(dsize, dtype=torch.uint8, device="cuda")
    status, n = nj.nvjpeg_encode_retrieve_bitstream_device(handle, state, dev.data_ptr(), dsize, s)
    ok(status)
    stream.synchronize()
    assert n == dsize and dev.cpu().numpy().tobytes() == exact

    # Too-small buffer is an error status, not a crash.
    encode()
    small = np.empty(16, np.uint8)
    status, _ = nj.nvjpeg_encode_retrieve_bitstream(handle, state, small.ctypes.data, 16, s)
    assert status != S.SUCCESS


def test_copy_quantization_tables_and_metadata(handle, encoder, rgb, stream):
    """Re-encoding with the source JPEG's quant tables reproduces its quality."""
    state, params = encoder
    s = stream.cuda_stream
    h, w, _ = rgb.shape
    source = cv_encode(rgb, quality=30, sampling="444")
    status, js = nj.nvjpeg_jpeg_stream_create(handle)
    ok(status)
    ok(nj.nvjpeg_jpeg_stream_parse(handle, source, None, 1, 0, js))
    ok(nj.nvjpeg_encoder_params_copy_quantization_tables(params, js, s))
    ok(nj.nvjpeg_encoder_params_copy_metadata(state, params, js, s))
    t = torch.from_numpy(rgb).cuda()
    torch.cuda.synchronize()
    ok(nj.nvjpeg_encode_image(handle, state, params, interleaved_image(t), IF.RGBI, w, h, s))
    out = retrieve(handle, state, stream)
    ok(nj.nvjpeg_jpeg_stream_destroy(js))

    def dqt(j):
        i, tables = 2, []
        while j[i + 1] != 0xDA:
            n = int.from_bytes(j[i + 2 : i + 4], "big")
            if j[i + 1] == 0xDB:
                tables.append(j[i + 4 : i + 2 + n])
            i += 2 + n
        return b"".join(tables)

    # Identical quantization tables (same bytes, same order/ids).
    assert dqt(out) == dqt(source)


def test_hardware_encoder_backend(handle, stream):
    status, engines = nj.nvjpeg_get_hardware_encoder_info(handle)
    ok(status)
    status, state = nj.nvjpeg_encoder_state_create_with_backend(handle, nj.nvjpegEncBackend_t.HARDWARE, stream.cuda_stream)
    if engines == 0:
        assert status != S.SUCCESS and not state
    else:
        ok(status)
        ok(nj.nvjpeg_encoder_state_destroy(state))
    status, state = nj.nvjpeg_encoder_state_create_with_backend(handle, nj.nvjpegEncBackend_t.GPU, stream.cuda_stream)
    ok(status)
    ok(nj.nvjpeg_encoder_state_destroy(state))


def test_unset_sampling_factors_is_invalid(handle, stream):
    """Documented gotcha: fresh params need sampling factors before encoding."""
    s = stream.cuda_stream
    _, state = nj.nvjpeg_encoder_state_create(handle, s)
    _, params = nj.nvjpeg_encoder_params_create(handle, s)
    t = torch.zeros((64, 64, 3), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    status = nj.nvjpeg_encode_image(handle, state, params, interleaved_image(t), IF.RGBI, 64, 64, s)
    assert status == S.INVALID_PARAMETER
    nj.nvjpeg_encoder_params_destroy(params)
    nj.nvjpeg_encoder_state_destroy(state)
