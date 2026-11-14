import cv2
import numpy as np
import pytest
import torch

import pynvjpeg as nj

S = nj.nvjpegStatus_t

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)


def ok(status):
    assert status == S.SUCCESS, status


def synthetic_rgb(h=480, w=640, seed=0):
    """Smooth gradients plus mild noise: JPEG-friendly and non-trivial."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack(
        [x * 255.0 / w, y * 255.0 / h, (x + y) * 127.0 / (w + h) + 64], axis=-1
    )
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


_CV_SAMPLING = {
    "444": cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
    "422": cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422,
    "420": cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
}


def cv_encode(rgb, quality=95, sampling="420", progressive=False):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if rgb.ndim == 3:
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, _CV_SAMPLING[sampling]]
        src = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        src = rgb
    if progressive:
        params += [cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
    ok_, buf = cv2.imencode(".jpg", src, params)
    assert ok_
    return buf.tobytes()


def cv_decode_rgb(jpeg):
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def interleaved_image(t):
    """NvjpegImage over an HxWxC uint8 CUDA tensor (channel 0 only)."""
    h, w, c = t.shape
    return nj.NvjpegImage([t.data_ptr()], [w * c])


def planar_image(t):
    """NvjpegImage over a CxHxW uint8 CUDA tensor (one channel per plane)."""
    c, h, w = t.shape
    return nj.NvjpegImage([t[i].data_ptr() for i in range(c)], [w] * c)


@pytest.fixture(scope="session")
def handle():
    status, h = nj.nvjpeg_create_simple()
    ok(status)
    yield h
    ok(nj.nvjpeg_destroy(h))


@pytest.fixture
def stream():
    s = torch.cuda.Stream()
    return s


@pytest.fixture(scope="session")
def rgb():
    return synthetic_rgb()
