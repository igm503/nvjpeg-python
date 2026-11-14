# nvjpeg-python

`pynvjpeg` is a transparent 1:1 Python binding of the NVIDIA nvJPEG C API. Use it to
write low-level nvJPEG code without writing C++.

There are no wrapper classes and no hidden synchronization or allocation. Every function
in `nvjpeg.h` is bound, and `tests/test_api.py::test_every_header_function_is_bound`
checks the header against the module.

## Build / install

Requirements: the CUDA toolkit (headers + `libnvjpeg`; no `nvcc` needed),
CMake ≥ 3.18, a C++17 compiler, and Python ≥ 3.9 with `nanobind`.

```bash
pip install .                       # scikit-build-core; installs pynvjpeg + pynvjpeg.pyi
# or, for development:
cmake -S . -B build && cmake --build build -j   # -> build/pynvjpeg*.so, build/pynvjpeg.pyi
PYTHONPATH=build python -m pytest tests          # needs a GPU, torch, numpy, opencv-python
```

## Conventions

| C | Python |
|---|---|
| `nvjpegFooBar(...)` | `nvjpeg_foo_bar(...)`, same argument order |
| output pointer args (`nvjpegHandle_t*`, `size_t*`, …) | dropped from the arguments and returned: `(status, out1, out2, ...)` |
| in/out args (`length` in `nvjpegEncodeRetrieveBitstream`) | passed in **and** returned |
| no outputs | returns `nvjpegStatus_t` |
| opaque handles | `NvjpegHandle`, `NvjpegJpegState`, `NvjpegEncoderState`, `NvjpegEncoderParams`, `NvjpegJpegStream`, `NvjpegDecodeParams`, `NvjpegJpegDecoder`, `NvjpegBufferPinned`, `NvjpegBufferDevice` (`.ptr` gives the raw address) |
| `nvjpegImage_t` | `NvjpegImage(channel=[ptr, ...], pitch=[bytes_per_row, ...])`, with `.channel` / `.pitch` read-write |
| `cudaStream_t`, device/host pointers | `int` (e.g. `torch.cuda.Stream().cuda_stream`, `tensor.data_ptr()`); `stream` defaults to `0` |
| `const unsigned char* data, size_t length` (host JPEG bytes) | `data` is any contiguous buffer (`bytes`, `bytearray`, `memoryview`, numpy) with `length=None`, or a raw `int` pointer plus `length` |
| arrays of bitstreams / images / params (batched) | Python lists |
| allocator structs | `NvjpegDevAllocator`, `NvjpegPinnedAllocator`, `…V2`: fields are C function-pointer addresses (from `ctypes`); pass `None` for the default |
| enums / `#define`s | `nvjpegStatus_t.SUCCESS`, `nvjpegOutputFormat_t.RGBI`, `NVJPEG_FLAGS_ENABLE_MEMORY_POOLS`, … |

Errors are **status codes**, as in C. Python exceptions are raised only for misuse
that C can't express: a wrong type, a channel index out of range, mismatched list
lengths, or a pointer without `length`.

Lifetimes are explicit. Call the matching `*_destroy`, which resets the object's
`.ptr` to 0. Destroying a NULL object returns `INVALID_PARAMETER`; it doesn't crash.

The GIL is released for every nvJPEG call, so several Python threads (each with
their own state, as nvJPEG requires) run concurrently.

## Example: decode, then re-encode

```python
import numpy as np, torch, pynvjpeg as nj

S, OF, IF = nj.nvjpegStatus_t, nj.nvjpegOutputFormat_t, nj.nvjpegInputFormat_t
stream = torch.cuda.Stream(); s = stream.cuda_stream
jpeg = open("photo.jpg", "rb").read()

status, handle = nj.nvjpeg_create_simple()
status, n, subsampling, widths, heights = nj.nvjpeg_get_image_info(handle, jpeg)
w, h = widths[0], heights[0]

# decode -> HxWx3 RGB tensor
status, state = nj.nvjpeg_jpeg_state_create(handle)
rgb = torch.empty((h, w, 3), dtype=torch.uint8, device="cuda")
status = nj.nvjpeg_decode(handle, state, jpeg, None, OF.RGBI,
                          nj.NvjpegImage([rgb.data_ptr()], [w * 3]), s)

# encode it back
status, enc_state = nj.nvjpeg_encoder_state_create(handle, s)
status, params = nj.nvjpeg_encoder_params_create(handle, s)
nj.nvjpeg_encoder_params_set_quality(params, 90, s)
nj.nvjpeg_encoder_params_set_sampling_factors(params, nj.nvjpegChromaSubsampling_t.CSS_420, s)
status = nj.nvjpeg_encode_image(handle, enc_state, params,
                                nj.NvjpegImage([rgb.data_ptr()], [w * 3]), IF.RGBI, w, h, s)
status, size = nj.nvjpeg_encode_retrieve_bitstream(handle, enc_state, 0, 0, s)   # size query
out = np.empty(size, np.uint8)
status, size = nj.nvjpeg_encode_retrieve_bitstream(handle, enc_state, out.ctypes.data, size, s)
stream.synchronize()

for destroy, obj in [(nj.nvjpeg_encoder_params_destroy, params),
                     (nj.nvjpeg_encoder_state_destroy, enc_state),
                     (nj.nvjpeg_jpeg_state_destroy, state), (nj.nvjpeg_destroy, handle)]:
    destroy(obj)
```

`examples/roundtrip.py` is a runnable version. The tests show every other part of
the API: batched decode, the decoupled host/transfer/device phases with attached
pinned/device buffers, ROI decode, `nvjpegEncodeYUV`, NV12 via `nvjpegEncode`,
device-side bitstream retrieval, quant-table copying, and custom allocators.

## Warning

nvJPEG itself has some strange behaviors. If it's a problem in nvJPEG, it will be
a problem in nvjpeg-python. This library is not something I would recommend using
for normal projects, unless you could already do it in C++ but don't want to make
bindings yourself. Examples:
