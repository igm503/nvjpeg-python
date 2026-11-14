"""Decoding: legacy single-image, batched, and the decoupled (stream/decoder) API."""

import struct

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
    synthetic_rgb,
)

OF = nj.nvjpegOutputFormat_t
CSS = nj.nvjpegChromaSubsampling_t
B = nj.nvjpegBackend_t


@pytest.fixture
def jpeg_state(handle):
    status, st = nj.nvjpeg_jpeg_state_create(handle)
    ok(status)
    yield st
    ok(nj.nvjpeg_jpeg_state_destroy(st))
    assert not st


# --------------------------------------------------------------------------
# Image info
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sampling,css,cw,ch",
    [("444", CSS.CSS_444, 640, 480), ("422", CSS.CSS_422, 320, 480), ("420", CSS.CSS_420, 320, 240)],
)
def test_get_image_info(handle, rgb, sampling, css, cw, ch):
    status, n, sub, widths, heights = nj.nvjpeg_get_image_info(handle, cv_encode(rgb, sampling=sampling))
    ok(status)
    assert n == 3 and sub == css
    assert widths == [640, cw, cw, 0] and heights == [480, ch, ch, 0]


def test_get_image_info_gray(handle, rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    status, n, sub, widths, heights = nj.nvjpeg_get_image_info(handle, cv_encode(gray))
    ok(status)
    assert n == 1 and sub == CSS.CSS_GRAY and widths[0] == 640 and heights[0] == 480


def test_data_argument_forms(handle, rgb):
    """bytes, bytearray, memoryview, numpy, and raw pointer + length all work."""
    jpeg = cv_encode(rgb)
    arr = np.frombuffer(jpeg, np.uint8).copy()
    for data, length in [
        (jpeg, None),
        (bytearray(jpeg), None),
        (memoryview(jpeg), None),
        (arr, None),
        (arr.ctypes.data, arr.nbytes),
        (jpeg + b"trailing garbage", len(jpeg)),
    ]:
        status, n, *_ = nj.nvjpeg_get_image_info(handle, data, length)
        ok(status)
        assert n == 3
    with pytest.raises(ValueError):
        nj.nvjpeg_get_image_info(handle, arr.ctypes.data)  # pointer needs length
    with pytest.raises(ValueError):
        nj.nvjpeg_get_image_info(handle, jpeg, len(jpeg) + 1)
    with pytest.raises(TypeError):
        nj.nvjpeg_get_image_info(handle, "not a buffer")


def test_bad_jpeg_returns_status(handle, jpeg_state):
    for junk in (b"garbage", b"\xff\xd8 not really a jpeg"):
        status, *_ = nj.nvjpeg_get_image_info(handle, junk)
        assert status in (S.BAD_JPEG, S.JPEG_NOT_SUPPORTED)
    out = torch.empty((16, 16, 3), dtype=torch.uint8, device="cuda")
    status = nj.nvjpeg_decode(handle, jpeg_state, b"\x00" * 64, None, OF.RGBI, interleaved_image(out), 0)
    assert status != S.SUCCESS


# --------------------------------------------------------------------------
# nvjpegDecode
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", ["444", "422", "420"])
def test_decode_rgbi_matches_libjpeg(handle, jpeg_state, rgb, stream, sampling):
    jpeg = cv_encode(rgb, sampling=sampling)
    out = torch.empty(rgb.shape, dtype=torch.uint8, device="cuda")
    ok(nj.nvjpeg_decode(handle, jpeg_state, jpeg, None, OF.RGBI, interleaved_image(out), stream.cuda_stream))
    stream.synchronize()
    got = out.cpu().numpy()
    ref = cv_decode_rgb(jpeg)
    # Different IDCT / chroma upsampling than libjpeg: tiny differences only.
    assert psnr(got, ref) > 38
    assert psnr(got, rgb) > 30


def test_decode_planar_and_bgr(handle, jpeg_state, rgb, stream):
    jpeg = cv_encode(rgb, sampling="444")
    h, w, _ = rgb.shape
    planes = torch.empty((3, h, w), dtype=torch.uint8, device="cuda")
    ok(nj.nvjpeg_decode(handle, jpeg_state, jpeg, None, OF.RGB, planar_image(planes), stream.cuda_stream))
    bgri = torch.empty((h, w, 3), dtype=torch.uint8, device="cuda")
    ok(nj.nvjpeg_decode(handle, jpeg_state, jpeg, None, OF.BGRI, interleaved_image(bgri), stream.cuda_stream))
    stream.synchronize()
    planar_rgb = planes.permute(1, 2, 0).cpu().numpy()
    assert np.array_equal(planar_rgb, bgri.cpu().numpy()[..., ::-1])
    assert psnr(planar_rgb, cv_decode_rgb(jpeg)) > 38


def test_decode_y_and_yuv(handle, jpeg_state, rgb, stream):
    jpeg = cv_encode(rgb, sampling="420")
    h, w, _ = rgb.shape
    y = torch.empty((h, w), dtype=torch.uint8, device="cuda")
    ok(nj.nvjpeg_decode(handle, jpeg_state, jpeg, None, OF.Y, nj.NvjpegImage([y.data_ptr()], [w]), stream.cuda_stream))
    yp = torch.empty((h, w), dtype=torch.uint8, device="cuda")
    u = torch.empty((h // 2, w // 2), dtype=torch.uint8, device="cuda")
    v = torch.empty((h // 2, w // 2), dtype=torch.uint8, device="cuda")
    img = nj.NvjpegImage([yp.data_ptr(), u.data_ptr(), v.data_ptr()], [w, w // 2, w // 2])
    ok(nj.nvjpeg_decode(handle, jpeg_state, jpeg, None, OF.YUV, img, stream.cuda_stream))
    stream.synchronize()
    assert torch.equal(y, yp)
    ref_y = cv2.cvtColor(cv_decode_rgb(jpeg), cv2.COLOR_RGB2YCrCb)[..., 0]
    assert np.abs(y.cpu().numpy().astype(int) - ref_y).mean() < 1.5


def test_decode_gray_jpeg(handle, jpeg_state, rgb, stream):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    jpeg = cv_encode(gray)
    out = torch.empty(gray.shape, dtype=torch.uint8, device="cuda")
    ok(nj.nvjpeg_decode(handle, jpeg_state, jpeg, None, OF.Y, nj.NvjpegImage([out.data_ptr()], [gray.shape[1]]), stream.cuda_stream))
    stream.synchronize()
    ref = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert psnr(out.cpu().numpy(), ref) > 40


def test_decode_progressive(handle, jpeg_state, rgb, stream):
    jpeg = cv_encode(rgb, progressive=True)
    out = torch.empty(rgb.shape, dtype=torch.uint8, device="cuda")
    ok(nj.nvjpeg_decode(handle, jpeg_state, jpeg, None, OF.RGBI, interleaved_image(out), stream.cuda_stream))
    stream.synchronize()
    assert psnr(out.cpu().numpy(), cv_decode_rgb(jpeg)) > 38


def test_decode_with_pitch_padding(handle, jpeg_state, rgb, stream):
    """Destination pitch larger than the row width is honoured."""
    h, w, _ = rgb.shape
    jpeg = cv_encode(rgb)
    pitch = w * 3 + 128
    buf = torch.zeros((h, pitch), dtype=torch.uint8, device="cuda")
    ok(nj.nvjpeg_decode(handle, jpeg_state, jpeg, None, OF.RGBI, nj.NvjpegImage([buf.data_ptr()], [pitch]), stream.cuda_stream))
    stream.synchronize()
    got = buf[:, : w * 3].reshape(h, w, 3).cpu().numpy()
    assert psnr(got, cv_decode_rgb(jpeg)) > 38
    assert int(buf[:, w * 3 :].abs().sum()) == 0


# --------------------------------------------------------------------------
# Batched (legacy)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pre_allocate", [False, True])
def test_decode_batched(handle, jpeg_state, stream, pre_allocate):
    imgs = [synthetic_rgb(240, 320, seed=i) for i in range(5)]
    jpegs = [cv_encode(im) for im in imgs]
    ok(nj.nvjpeg_decode_batched_initialize(handle, jpeg_state, len(jpegs), 2, OF.RGBI))
    if pre_allocate:
        ok(nj.nvjpeg_decode_batched_pre_allocate(handle, jpeg_state, len(jpegs), 320, 240, CSS.CSS_420, OF.RGBI))
    outs = [torch.empty((240, 320, 3), dtype=torch.uint8, device="cuda") for _ in jpegs]
    # Mix of buffer objects and raw pointers.
    arrs = [np.frombuffer(j, np.uint8).copy() for j in jpegs]
    data = [jpegs[0], bytearray(jpegs[1]), arrs[2], arrs[3].ctypes.data, arrs[4].ctypes.data]
    lengths = [len(j) for j in jpegs]
    ok(nj.nvjpeg_decode_batched(handle, jpeg_state, data, lengths, [interleaved_image(o) for o in outs], stream.cuda_stream))
    stream.synchronize()
    for o, j in zip(outs, jpegs):
        assert psnr(o.cpu().numpy(), cv_decode_rgb(j)) > 38


def split_tables(jpeg):
    """Split a JPEG into a tables-only stream (DQT/DHT) and an abbreviated image."""
    segments, i = [], 2
    while True:
        assert jpeg[i] == 0xFF
        marker = jpeg[i + 1]
        if marker == 0xDA:  # start of scan: the rest is entropy-coded data
            segments.append((marker, jpeg[i:]))
            break
        n = struct.unpack(">H", jpeg[i + 2 : i + 4])[0]
        segments.append((marker, jpeg[i : i + 2 + n]))
        i += 2 + n
    is_table = lambda m: m in (0xDB, 0xC4)
    tables = b"\xff\xd8" + b"".join(s for m, s in segments if is_table(m)) + b"\xff\xd9"
    image = b"\xff\xd8" + b"".join(s for m, s in segments if not is_table(m))
    return tables, image


def test_decode_batched_parse_jpeg_tables(handle, jpeg_state, rgb, stream):
    jpeg = cv_encode(rgb)
    tables, abbreviated = split_tables(jpeg)
    ok(nj.nvjpeg_decode_batched_initialize(handle, jpeg_state, 1, 1, OF.RGBI))
    ok(nj.nvjpeg_decode_batched_parse_jpeg_tables(handle, jpeg_state, tables))
    out = torch.empty(rgb.shape, dtype=torch.uint8, device="cuda")
    ok(nj.nvjpeg_decode_batched(handle, jpeg_state, [abbreviated], None, [interleaved_image(out)], stream.cuda_stream))
    stream.synchronize()
    assert psnr(out.cpu().numpy(), cv_decode_rgb(jpeg)) > 38


def test_decode_batched_length_mismatch(handle, jpeg_state):
    with pytest.raises(ValueError):
        nj.nvjpeg_decode_batched(handle, jpeg_state, [b"a", b"b"], [1], [nj.NvjpegImage(), nj.NvjpegImage()], 0)
    with pytest.raises(ValueError):
        nj.nvjpeg_decode_batched(handle, jpeg_state, [b"a", b"b"], None, [nj.NvjpegImage()], 0)


# --------------------------------------------------------------------------
# Decoupled API: jpeg stream + decoder + decode params
# --------------------------------------------------------------------------


@pytest.fixture
def jpeg_stream(handle):
    status, js = nj.nvjpeg_jpeg_stream_create(handle)
    ok(status)
    yield js
    ok(nj.nvjpeg_jpeg_stream_destroy(js))


@pytest.fixture
def decode_params(handle):
    status, p = nj.nvjpeg_decode_params_create(handle)
    ok(status)
    yield p
    ok(nj.nvjpeg_decode_params_destroy(p))


def test_jpeg_stream_queries(handle, jpeg_stream, rgb):
    jpeg = cv_encode(rgb, sampling="422")
    ok(nj.nvjpeg_jpeg_stream_parse_header(handle, jpeg, None, jpeg_stream))
    assert nj.nvjpeg_jpeg_stream_get_frame_dimensions(jpeg_stream) == (S.SUCCESS, 640, 480)
    ok(nj.nvjpeg_jpeg_stream_parse(handle, jpeg, None, 0, 1, jpeg_stream))
    assert nj.nvjpeg_jpeg_stream_get_components_num(jpeg_stream) == (S.SUCCESS, 3)
    assert nj.nvjpeg_jpeg_stream_get_component_dimensions(jpeg_stream, 0) == (S.SUCCESS, 640, 480)
    assert nj.nvjpeg_jpeg_stream_get_component_dimensions(jpeg_stream, 1) == (S.SUCCESS, 320, 480)
    assert nj.nvjpeg_jpeg_stream_get_chroma_subsampling(jpeg_stream) == (S.SUCCESS, CSS.CSS_422)
    assert nj.nvjpeg_jpeg_stream_get_sample_precision(jpeg_stream) == (S.SUCCESS, 8)
    status, enc = nj.nvjpeg_jpeg_stream_get_jpeg_encoding(jpeg_stream)
    ok(status)
    assert enc == nj.nvjpegJpegEncoding_t.BASELINE_DCT
    status, orient = nj.nvjpeg_jpeg_stream_get_exif_orientation(jpeg_stream)
    ok(status)
    assert orient in (nj.nvjpegExifOrientation_t.UNKNOWN, nj.nvjpegExifOrientation_t.NORMAL)

    ok(nj.nvjpeg_jpeg_stream_parse(handle, cv_encode(rgb, progressive=True), None, 0, 1, jpeg_stream))
    assert nj.nvjpeg_jpeg_stream_get_jpeg_encoding(jpeg_stream)[1] == nj.nvjpegJpegEncoding_t.PROGRESSIVE_DCT_HUFFMAN


def test_jpeg_stream_parse_tables_then_abbreviated(handle, jpeg_stream, decode_params, rgb, stream):
    dec, st, pb, db = _decoupled_setup(handle, B.DEFAULT)
    jpeg = cv_encode(rgb)
    tables, abbreviated = split_tables(jpeg)
    ok(nj.nvjpeg_jpeg_stream_parse_tables(handle, tables, None, jpeg_stream))
    ok(nj.nvjpeg_jpeg_stream_parse(handle, abbreviated, None, 0, 0, jpeg_stream))
    ok(nj.nvjpeg_decode_params_set_output_format(decode_params, OF.RGBI))
    out = torch.empty(rgb.shape, dtype=torch.uint8, device="cuda")
    ok(nj.nvjpeg_decode_jpeg(handle, dec, st, jpeg_stream, interleaved_image(out), decode_params, stream.cuda_stream))
    stream.synchronize()
    assert psnr(out.cpu().numpy(), cv_decode_rgb(jpeg)) > 38
    _decoupled_teardown(dec, st, pb, db)


def _decoupled_setup(handle, backend):
    status, dec = nj.nvjpeg_decoder_create(handle, backend)
    if status != S.SUCCESS:
        pytest.skip(f"decoder backend {backend} unavailable: {status}")
    status, st = nj.nvjpeg_decoder_state_create(handle, dec)
    ok(status)
    status, pb = nj.nvjpeg_buffer_pinned_create(handle, None)
    ok(status)
    status, db = nj.nvjpeg_buffer_device_create(handle, None)
    ok(status)
    ok(nj.nvjpeg_state_attach_pinned_buffer(st, pb))
    ok(nj.nvjpeg_state_attach_device_buffer(st, db))
    return dec, st, pb, db


def _decoupled_teardown(dec, st, pb, db):
    ok(nj.nvjpeg_jpeg_state_destroy(st))
    ok(nj.nvjpeg_buffer_pinned_destroy(pb))
    ok(nj.nvjpeg_buffer_device_destroy(db))
    ok(nj.nvjpeg_decoder_destroy(dec))


@pytest.mark.parametrize("backend", [B.DEFAULT, B.HYBRID, B.GPU_HYBRID, B.HARDWARE])
def test_decoupled_phases(handle, jpeg_stream, decode_params, rgb, stream, backend):
    dec, st, pb, db = _decoupled_setup(handle, backend)
    jpeg = cv_encode(rgb)
    ok(nj.nvjpeg_jpeg_stream_parse(handle, jpeg, None, 0, 0, jpeg_stream))
    ok(nj.nvjpeg_decode_params_set_output_format(decode_params, OF.RGBI))
    status, unsupported = nj.nvjpeg_decoder_jpeg_supported(dec, jpeg_stream, decode_params)
    ok(status)
    if unsupported:
        _decoupled_teardown(dec, st, pb, db)
        pytest.skip(f"{backend} does not support this stream")
    out = torch.empty(rgb.shape, dtype=torch.uint8, device="cuda")
    s = stream.cuda_stream
    ok(nj.nvjpeg_decode_jpeg_host(handle, dec, st, decode_params, jpeg_stream))
    ok(nj.nvjpeg_decode_jpeg_transfer_to_device(handle, dec, st, jpeg_stream, s))
    ok(nj.nvjpeg_decode_jpeg_device(handle, dec, st, interleaved_image(out), s))
    stream.synchronize()
    assert psnr(out.cpu().numpy(), cv_decode_rgb(jpeg)) > 38
    _decoupled_teardown(dec, st, pb, db)


def test_decode_jpeg_roi(handle, jpeg_stream, decode_params, rgb, stream):
    dec, st, pb, db = _decoupled_setup(handle, B.DEFAULT)
    jpeg = cv_encode(rgb)
    x, y, w, h = 64, 32, 256, 128
    ok(nj.nvjpeg_jpeg_stream_parse(handle, jpeg, None, 0, 0, jpeg_stream))
    ok(nj.nvjpeg_decode_params_set_output_format(decode_params, OF.RGBI))
    ok(nj.nvjpeg_decode_params_set_roi(decode_params, x, y, w, h))
    out = torch.empty((h, w, 3), dtype=torch.uint8, device="cuda")
    ok(nj.nvjpeg_decode_jpeg(handle, dec, st, jpeg_stream, interleaved_image(out), decode_params, stream.cuda_stream))
    stream.synchronize()
    ref = cv_decode_rgb(jpeg)[y : y + h, x : x + w]
    assert psnr(out.cpu().numpy(), ref) > 38
    _decoupled_teardown(dec, st, pb, db)


def test_decode_params_setters(decode_params):
    ok(nj.nvjpeg_decode_params_set_allow_cmyk(decode_params, 1))
    ok(nj.nvjpeg_decode_params_set_scale_factor(decode_params, nj.nvjpegScaleFactor_t.SCALE_NONE))
    ok(nj.nvjpeg_decode_params_set_exif_orientation(decode_params, nj.nvjpegExifOrientation_t.NORMAL))


def test_decode_batched_ex(handle, stream):
    """Batched decode with per-image params (hardware backend when present)."""
    backend = B.HARDWARE
    status, hw = nj.nvjpeg_create_ex(backend, None, None, 0)
    if status != S.SUCCESS:
        backend = B.DEFAULT
        status, hw = nj.nvjpeg_create_ex(backend, None, None, 0)
    ok(status)
    status, st = nj.nvjpeg_jpeg_state_create(hw)
    ok(status)
    imgs = [synthetic_rgb(240, 320, seed=i) for i in range(4)]
    jpegs = [cv_encode(im) for im in imgs]

    status, js = nj.nvjpeg_jpeg_stream_create(hw)
    ok(status)
    ok(nj.nvjpeg_jpeg_stream_parse(hw, jpegs[0], None, 0, 0, js))
    status, unsupported = nj.nvjpeg_decode_batched_supported(hw, js)
    ok(status)

    params = []
    for _ in jpegs:
        status, p = nj.nvjpeg_decode_params_create(hw)
        ok(status)
        ok(nj.nvjpeg_decode_params_set_output_format(p, OF.RGBI))
        params.append(p)
    status, unsupported_ex = nj.nvjpeg_decode_batched_supported_ex(hw, js, params[0])
    ok(status)
    assert unsupported_ex == unsupported

    ok(nj.nvjpeg_decode_batched_initialize(hw, st, len(jpegs), 1, OF.RGBI))
    outs = [torch.empty((240, 320, 3), dtype=torch.uint8, device="cuda") for _ in jpegs]
    ok(nj.nvjpeg_decode_batched_ex(hw, st, jpegs, None, [interleaved_image(o) for o in outs], params, stream.cuda_stream))
    stream.synchronize()
    for o, j in zip(outs, jpegs):
        assert psnr(o.cpu().numpy(), cv_decode_rgb(j)) > 38

    for p in params:
        ok(nj.nvjpeg_decode_params_destroy(p))
    ok(nj.nvjpeg_jpeg_stream_destroy(js))
    ok(nj.nvjpeg_jpeg_state_destroy(st))
    ok(nj.nvjpeg_destroy(hw))
