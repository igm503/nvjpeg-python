"""Library-level API: coverage of nvjpeg.h, versions, handles, allocators, types."""

import ctypes
import ctypes.util
import os
import re
import threading
import time

import pytest
import torch

import pynvjpeg as nj
from conftest import S, ok

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")


def _header_functions():
    for path in (
        os.path.join(CUDA_HOME, "include", "nvjpeg.h"),
        "/usr/local/cuda/include/nvjpeg.h",
    ):
        if os.path.exists(path):
            text = open(path).read()
            return sorted(set(re.findall(r"nvjpegStatus_t\s+(?:NVJPEGAPI\s+)?(nvjpeg\w+)\s*\(", text)))
    pytest.skip("nvjpeg.h not found")


def _snake(name):
    # nvjpegDecodeParamsSetROI -> nvjpeg_decode_params_set_roi
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return s.lower()


def test_every_header_function_is_bound():
    funcs = _header_functions()
    assert len(funcs) >= 79
    missing = [f for f in funcs if not hasattr(nj, _snake(f))]
    assert not missing, missing


def test_versions_match_header():
    for prop, expected in [
        (nj.libraryPropertyType.MAJOR_VERSION, nj.NVJPEG_VER_MAJOR),
        (nj.libraryPropertyType.MINOR_VERSION, nj.NVJPEG_VER_MINOR),
        (nj.libraryPropertyType.PATCH_LEVEL, nj.NVJPEG_VER_PATCH),
    ]:
        status, value = nj.nvjpeg_get_property(prop)
        ok(status)
        assert value == expected
    status, major = nj.nvjpeg_get_cudart_property(nj.libraryPropertyType.MAJOR_VERSION)
    ok(status)
    assert major >= 11


@pytest.mark.parametrize("backend", [nj.nvjpegBackend_t.DEFAULT, nj.nvjpegBackend_t.GPU_HYBRID])
def test_create_destroy(backend):
    status, h = nj.nvjpeg_create(backend)
    ok(status)
    assert h and h.ptr != 0
    ok(nj.nvjpeg_destroy(h))
    assert not h and h.ptr == 0
    # Destroying a NULL handle reports an error status instead of crashing.
    assert nj.nvjpeg_destroy(h) == S.INVALID_PARAMETER


def test_create_ex_flags():
    status, h = nj.nvjpeg_create_ex(
        nj.nvjpegBackend_t.DEFAULT, None, None, nj.NVJPEG_FLAGS_ENABLE_MEMORY_POOLS
    )
    ok(status)
    ok(nj.nvjpeg_destroy(h))


def _cudart():
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        for d in (os.path.join(CUDA_HOME, "lib64"), ""):
            try:
                return ctypes.CDLL(os.path.join(d, name) if d else name)
            except OSError:
                continue
    pytest.skip("libcudart not loadable")


def _addr(fn):
    return ctypes.cast(fn, ctypes.c_void_p).value


def test_create_ex_with_c_allocators(rgb):
    """Allocator structs take raw C function pointers (here: cudart's own)."""
    from conftest import cv_encode

    rt = _cudart()
    dev = nj.NvjpegDevAllocator(_addr(rt.cudaMalloc), _addr(rt.cudaFree))
    pin = nj.NvjpegPinnedAllocator(_addr(rt.cudaHostAlloc), _addr(rt.cudaFreeHost))
    assert dev.dev_malloc == _addr(rt.cudaMalloc)
    status, h = nj.nvjpeg_create_ex(nj.nvjpegBackend_t.DEFAULT, dev, pin, 0)
    ok(status)
    # Exercise the allocators with a decode.
    status, st = nj.nvjpeg_jpeg_state_create(h)
    ok(status)
    out = torch.empty(rgb.shape, dtype=torch.uint8, device="cuda")
    img = nj.NvjpegImage([out.data_ptr()], [rgb.shape[1] * 3])
    ok(nj.nvjpeg_decode(h, st, cv_encode(rgb), None, nj.nvjpegOutputFormat_t.RGBI, img, 0))
    torch.cuda.synchronize()
    ok(nj.nvjpeg_jpeg_state_destroy(st))
    ok(nj.nvjpeg_destroy(h))


def _cudart_typed():
    rt = _cudart()
    rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    rt.cudaFree.argtypes = [ctypes.c_void_p]
    rt.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    rt.cudaFreeHost.argtypes = [ctypes.c_void_p]
    return rt


V2_MALLOC = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_void_p
)
V2_FREE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
V1_DEV_MALLOC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
V1_PIN_MALLOC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint)
V1_FREE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class CountingAllocators:
    """ctypes callbacks around cudart that record calls (and V2 contexts)."""

    def __init__(self):
        rt = _cudart_typed()
        self.calls = {"dev_malloc": 0, "dev_free": 0, "pin_malloc": 0, "pin_free": 0}
        self.ctx = set()

        def count(name, ctx=None):
            self.calls[name] += 1
            if ctx is not None:
                self.ctx.add(ctx)

        # Keep references: the C side only stores raw function pointers.
        self.v1_dev = (
            V1_DEV_MALLOC(lambda p, n: (count("dev_malloc"), rt.cudaMalloc(p, n))[1]),
            V1_FREE(lambda p: (count("dev_free"), rt.cudaFree(p))[1]),
        )
        self.v1_pin = (
            V1_PIN_MALLOC(lambda p, n, f: (count("pin_malloc"), rt.cudaHostAlloc(p, n, f))[1]),
            V1_FREE(lambda p: (count("pin_free"), rt.cudaFreeHost(p))[1]),
        )
        self.v2_dev = (
            V2_MALLOC(lambda c, p, n, s: (count("dev_malloc", c), rt.cudaMalloc(p, n))[1]),
            V2_FREE(lambda c, p, n, s: (count("dev_free", c), rt.cudaFree(p))[1]),
        )
        self.v2_pin = (
            V2_MALLOC(lambda c, p, n, s: (count("pin_malloc", c), rt.cudaHostAlloc(p, n, 0))[1]),
            V2_FREE(lambda c, p, n, s: (count("pin_free", c), rt.cudaFreeHost(p))[1]),
        )

    def dev(self):
        return nj.NvjpegDevAllocator(*map(_addr, self.v1_dev))

    def pin(self):
        return nj.NvjpegPinnedAllocator(*map(_addr, self.v1_pin))

    def dev_v2(self, ctx):
        return nj.NvjpegDevAllocatorV2(*map(_addr, self.v2_dev), ctx)

    def pin_v2(self, ctx):
        return nj.NvjpegPinnedAllocatorV2(*map(_addr, self.v2_pin), ctx)


def test_create_ex_v2_with_callbacks(rgb):
    from conftest import cv_encode

    alloc = CountingAllocators()
    status, h = nj.nvjpeg_create_ex_v2(nj.nvjpegBackend_t.DEFAULT, alloc.dev_v2(0xD0), alloc.pin_v2(0xD1), 0)
    ok(status)
    status, st = nj.nvjpeg_jpeg_state_create(h)
    ok(status)
    out = torch.empty(rgb.shape, dtype=torch.uint8, device="cuda")
    img = nj.NvjpegImage([out.data_ptr()], [rgb.shape[1] * 3])
    ok(nj.nvjpeg_decode(h, st, cv_encode(rgb), None, nj.nvjpegOutputFormat_t.RGBI, img, 0))
    torch.cuda.synchronize()
    ok(nj.nvjpeg_jpeg_state_destroy(st))
    ok(nj.nvjpeg_destroy(h))
    # Note: nvJPEG 13 does not route nvjpegDecode's internal allocations through
    # handle-level allocators, so no call count is asserted here; the buffer
    # tests below verify the callback plumbing.
    assert alloc.ctx <= {0xD0, 0xD1}


def test_memory_padding(handle):
    ok(nj.nvjpeg_set_device_memory_padding(1024, handle))
    status, padding = nj.nvjpeg_get_device_memory_padding(handle)
    ok(status)
    assert padding == 1024
    ok(nj.nvjpeg_set_pinned_memory_padding(2048, handle))
    status, padding = nj.nvjpeg_get_pinned_memory_padding(handle)
    ok(status)
    assert padding == 2048
    ok(nj.nvjpeg_set_device_memory_padding(0, handle))
    ok(nj.nvjpeg_set_pinned_memory_padding(0, handle))


def test_hardware_info(handle):
    status, engines, cores = nj.nvjpeg_get_hardware_decoder_info(handle)
    ok(status)
    assert engines >= 0 and cores >= 0
    status, enc_engines = nj.nvjpeg_get_hardware_encoder_info(handle)
    ok(status)
    assert enc_engines >= 0


def test_nvjpeg_image_fields():
    img = nj.NvjpegImage()
    assert img.channel == [0] * nj.MAX_COMPONENT
    assert img.pitch == [0] * nj.MAX_COMPONENT
    img = nj.NvjpegImage([0x1000, 0x2000], [64, 32])
    assert img.channel[:2] == [0x1000, 0x2000] and img.channel[2:] == [0, 0]
    assert img.pitch[:2] == [64, 32]
    nj.nvjpeg_image_set_channel(img, 3, 0x3000, 16)
    assert nj.nvjpeg_image_get_channel_ptr(img, 3) == 0x3000
    assert nj.nvjpeg_image_get_pitch(img, 3) == 16
    img.pitch = [1, 2, 3]
    assert img.pitch == [1, 2, 3, 0]
    with pytest.raises(IndexError):
        nj.nvjpeg_image_set_channel(img, nj.MAX_COMPONENT, 0, 0)
    with pytest.raises(IndexError):
        nj.nvjpeg_image_get_pitch(img, -1)
    with pytest.raises(ValueError):
        nj.NvjpegImage([1, 2], [1])


def test_opaque_types_default_to_null():
    for cls in (
        nj.NvjpegHandle,
        nj.NvjpegJpegState,
        nj.NvjpegEncoderState,
        nj.NvjpegEncoderParams,
        nj.NvjpegBufferPinned,
        nj.NvjpegBufferDevice,
        nj.NvjpegJpegStream,
        nj.NvjpegDecodeParams,
        nj.NvjpegJpegDecoder,
    ):
        o = cls()
        assert o.ptr == 0 and not o
        assert "ptr=0x0" in repr(o)


def test_pinned_and_device_buffers(handle, stream):
    status, pb = nj.nvjpeg_buffer_pinned_create(handle, None)
    ok(status)
    ok(nj.nvjpeg_buffer_pinned_resize(pb, 1 << 20, stream.cuda_stream))
    status, size, ptr = nj.nvjpeg_buffer_pinned_retrieve(pb)
    ok(status)
    assert size >= 1 << 20 and ptr != 0
    ctypes.memset(ptr, 7, 16)  # really host-accessible memory

    status, db = nj.nvjpeg_buffer_device_create(handle, None)
    ok(status)
    ok(nj.nvjpeg_buffer_device_resize(db, 1 << 20, stream.cuda_stream))
    status, size, ptr = nj.nvjpeg_buffer_device_retrieve(db)
    ok(status)
    assert size >= 1 << 20 and ptr != 0
    stream.synchronize()

    ok(nj.nvjpeg_buffer_pinned_destroy(pb))
    ok(nj.nvjpeg_buffer_device_destroy(db))
    assert not pb and not db


@pytest.mark.parametrize("version", [1, 2])
def test_buffer_allocators_are_called(handle, version):
    alloc = CountingAllocators()
    if version == 1:
        status, db = nj.nvjpeg_buffer_device_create(handle, alloc.dev())
        ok(status)
        status, pb = nj.nvjpeg_buffer_pinned_create(handle, alloc.pin())
    else:
        status, db = nj.nvjpeg_buffer_device_create_v2(handle, alloc.dev_v2(0xABC))
        ok(status)
        status, pb = nj.nvjpeg_buffer_pinned_create_v2(handle, alloc.pin_v2(0xDEF))
    ok(status)
    ok(nj.nvjpeg_buffer_device_resize(db, 4096, 0))
    ok(nj.nvjpeg_buffer_pinned_resize(pb, 4096, 0))
    torch.cuda.synchronize()
    assert nj.nvjpeg_buffer_device_retrieve(db)[2] != 0
    assert nj.nvjpeg_buffer_pinned_retrieve(pb)[2] != 0
    assert alloc.calls["dev_malloc"] == 1 and alloc.calls["pin_malloc"] == 1
    ok(nj.nvjpeg_buffer_device_destroy(db))
    ok(nj.nvjpeg_buffer_pinned_destroy(pb))
    assert alloc.calls == {"dev_malloc": 1, "dev_free": 1, "pin_malloc": 1, "pin_free": 1}
    if version == 2:
        assert alloc.ctx == {0xABC, 0xDEF}


def test_gil_released_during_calls(handle):
    """While one thread is blocked inside nvJPEG, other Python threads run.

    nvjpegEncodeImage waits on its stream; a GPU sleep kernel queued first keeps
    it blocked for a while. If the binding held the GIL, the main thread could
    not run during that time.
    """
    s = torch.cuda.Stream()
    sp = s.cuda_stream
    ok_state, es = nj.nvjpeg_encoder_state_create(handle, sp)
    ok(ok_state)
    status, ep = nj.nvjpeg_encoder_params_create(handle, sp)
    ok(status)
    ok(nj.nvjpeg_encoder_params_set_sampling_factors(ep, nj.nvjpegChromaSubsampling_t.CSS_420, sp))
    x = torch.zeros((256, 256, 3), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    img = nj.NvjpegImage([x.data_ptr()], [256 * 3])

    result = {}

    def worker():
        result["status"] = nj.nvjpeg_encode_image(
            handle, es, ep, img, nj.nvjpegInputFormat_t.RGBI, 256, 256, sp
        )
        result["t_end"] = time.perf_counter()

    with torch.cuda.stream(s):
        torch.cuda._sleep(1_500_000_000)  # ~0.5 s of GPU time
    t = threading.Thread(target=worker)
    t.start()
    ticks = 0
    t_last = time.perf_counter()
    while t.is_alive():
        ticks += 1
        t_last = time.perf_counter()
    t.join()
    ok(result["status"])
    # The main thread kept executing Python well before the worker returned.
    assert ticks > 1000
    assert result["t_end"] - t_last < 0.05
    ok(nj.nvjpeg_encoder_params_destroy(ep))
    ok(nj.nvjpeg_encoder_state_destroy(es))
