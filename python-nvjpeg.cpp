// pynvjpeg: a transparent, 1:1 Python binding of the NVIDIA nvJPEG C API.
//
// Conventions (applied to every function):
//   * Each C function `nvjpegFooBar(...)` is exposed as `nvjpeg_foo_bar(...)`.
//   * Arguments keep the C order. Pure output pointers are dropped from the
//     argument list and returned instead: a function returns the bare
//     `nvjpegStatus_t` when it has no outputs, otherwise a tuple
//     `(status, out1, out2, ...)`. In/out arguments (e.g. `length` in
//     nvjpegEncodeRetrieveBitstream) are passed in and returned.
//   * Opaque handles (nvjpegHandle_t, nvjpegJpegState_t, ...) are small Python
//     objects. Their lifetime is managed explicitly, exactly as in C: call the
//     matching *_destroy function. Destroying resets the object's pointer to
//     NULL. `.ptr` exposes the raw handle address.
//   * cudaStream_t and device/host memory pointers are plain ints (e.g.
//     `torch.cuda.Stream().cuda_stream`, `tensor.data_ptr()`). 0 is NULL /
//     the default stream.
//   * Host JPEG bitstreams (`data`) may be any contiguous buffer-protocol object
//     (bytes, bytearray, memoryview, numpy array) or a raw int pointer; with a
//     pointer, `length` is required.
//   * The GIL is released while nvJPEG runs, so calls from multiple Python
//     threads proceed concurrently.

#include <cuda_runtime.h>
#include <library_types.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nvjpeg.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

using Release = nb::call_guard<nb::gil_scoped_release>;

namespace {

cudaStream_t as_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }
unsigned char *as_bytes(uintptr_t p) { return reinterpret_cast<unsigned char *>(p); }

// ---------------------------------------------------------------------------
// Opaque handle wrappers
// ---------------------------------------------------------------------------

struct NvjpegHandle { nvjpegHandle_t handle = nullptr; };
struct NvjpegJpegState { nvjpegJpegState_t state = nullptr; };
struct NvjpegEncoderState { nvjpegEncoderState_t state = nullptr; };
struct NvjpegEncoderParams { nvjpegEncoderParams_t params = nullptr; };
struct NvjpegBufferPinned { nvjpegBufferPinned_t buffer = nullptr; };
struct NvjpegBufferDevice { nvjpegBufferDevice_t buffer = nullptr; };
struct NvjpegJpegStream { nvjpegJpegStream_t stream = nullptr; };
struct NvjpegDecodeParams { nvjpegDecodeParams_t params = nullptr; };
struct NvjpegJpegDecoder { nvjpegJpegDecoder_t decoder = nullptr; };

struct NvjpegImage {
  nvjpegImage_t image{};
};

// Allocator structs: function pointers (and V2 contexts) are raw addresses,
// e.g. from ctypes: ctypes.cast(libcudart.cudaMalloc, ctypes.c_void_p).value
// or the address of a ctypes.CFUNCTYPE callback (which must be kept alive).
struct NvjpegDevAllocator {
  uintptr_t dev_malloc = 0, dev_free = 0;
  nvjpegDevAllocator_t get() const {
    return {reinterpret_cast<tDevMalloc>(dev_malloc),
            reinterpret_cast<tDevFree>(dev_free)};
  }
};
struct NvjpegPinnedAllocator {
  uintptr_t pinned_malloc = 0, pinned_free = 0;
  nvjpegPinnedAllocator_t get() const {
    return {reinterpret_cast<tPinnedMalloc>(pinned_malloc),
            reinterpret_cast<tPinnedFree>(pinned_free)};
  }
};
struct NvjpegDevAllocatorV2 {
  uintptr_t dev_malloc = 0, dev_free = 0, dev_ctx = 0;
  nvjpegDevAllocatorV2_t get() const {
    return {reinterpret_cast<tDevMallocV2>(dev_malloc),
            reinterpret_cast<tDevFreeV2>(dev_free),
            reinterpret_cast<void *>(dev_ctx)};
  }
};
struct NvjpegPinnedAllocatorV2 {
  uintptr_t pinned_malloc = 0, pinned_free = 0, pinned_ctx = 0;
  nvjpegPinnedAllocatorV2_t get() const {
    return {reinterpret_cast<tPinnedMallocV2>(pinned_malloc),
            reinterpret_cast<tPinnedFreeV2>(pinned_free),
            reinterpret_cast<void *>(pinned_ctx)};
  }
};

// Optional allocator argument -> C pointer (nullptr when None).
template <typename Py, typename C> struct OptAlloc {
  C value{};
  C *ptr = nullptr;
  explicit OptAlloc(const Py *a) {
    if (a) {
      value = a->get();
      ptr = &value;
    }
  }
};

// ---------------------------------------------------------------------------
// Host data argument: buffer-protocol object or raw pointer
// ---------------------------------------------------------------------------

class HostData {
public:
  HostData(nb::handle obj, std::optional<size_t> length) {
    if (PyLong_Check(obj.ptr())) {
      if (!length)
        throw nb::value_error("`length` is required when `data` is a pointer");
      ptr_ = as_bytes(nb::cast<uintptr_t>(obj));
      len_ = *length;
      return;
    }
    if (PyObject_GetBuffer(obj.ptr(), &view_, PyBUF_C_CONTIGUOUS) != 0)
      throw nb::python_error();
    has_view_ = true;
    ptr_ = static_cast<const unsigned char *>(view_.buf);
    size_t n = static_cast<size_t>(view_.len);
    if (length && *length > n) {
      PyBuffer_Release(&view_);
      has_view_ = false;
      throw nb::value_error("`length` exceeds the size of `data`");
    }
    len_ = length ? *length : n;
  }
  HostData(const HostData &) = delete;
  HostData &operator=(const HostData &) = delete;
  HostData(HostData &&o) noexcept
      : view_(o.view_), has_view_(o.has_view_), ptr_(o.ptr_), len_(o.len_) {
    o.has_view_ = false;
  }
  ~HostData() {
    if (has_view_)
      PyBuffer_Release(&view_);  // always destroyed with the GIL held
  }
  const unsigned char *ptr() const { return ptr_; }
  size_t len() const { return len_; }

private:
  Py_buffer view_{};
  bool has_view_ = false;
  const unsigned char *ptr_ = nullptr;
  size_t len_ = 0;
};

std::vector<HostData> host_data_list(const nb::sequence &data,
                                     const std::optional<std::vector<size_t>> &lengths) {
  size_t n = nb::len(data);
  if (lengths && lengths->size() != n)
    throw nb::value_error("`lengths` must have the same length as `data`");
  std::vector<HostData> out;
  out.reserve(n);
  for (size_t i = 0; i < n; ++i)
    out.emplace_back(data[i], lengths ? std::optional<size_t>((*lengths)[i])
                                      : std::nullopt);
  return out;
}

void check_channel(int idx) {
  if (idx < 0 || idx >= NVJPEG_MAX_COMPONENT)
    throw nb::index_error("channel index out of range [0, NVJPEG_MAX_COMPONENT)");
}

template <typename T, typename F>
void def_opaque(nb::module_ &m, const char *name, F member, const char *doc) {
  nb::class_<T>(m, name, doc)
      .def(nb::init<>())
      .def_prop_ro(
          "ptr",
          [member](const T &o) { return reinterpret_cast<uintptr_t>(o.*member); },
          "Raw handle address (0 when uninitialized or destroyed).")
      .def("__bool__", [member](const T &o) { return o.*member != nullptr; })
      .def("__repr__", [name, member](const T &o) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "<pynvjpeg.%s ptr=0x%zx>", name,
                      static_cast<size_t>(reinterpret_cast<uintptr_t>(o.*member)));
        return std::string(buf);
      });
}

}  // namespace

NB_MODULE(pynvjpeg, m) {
  m.doc() = "A transparent 1:1 Python binding of the NVIDIA nvJPEG C API.";

  // -------------------------------------------------------------------------
  // Constants
  // -------------------------------------------------------------------------
  m.attr("MAX_COMPONENT") = NVJPEG_MAX_COMPONENT;
  m.attr("NVJPEG_MAX_COMPONENT") = NVJPEG_MAX_COMPONENT;
  m.attr("NVJPEG_VER_MAJOR") = NVJPEG_VER_MAJOR;
  m.attr("NVJPEG_VER_MINOR") = NVJPEG_VER_MINOR;
  m.attr("NVJPEG_VER_PATCH") = NVJPEG_VER_PATCH;
  m.attr("NVJPEG_VER_BUILD") = NVJPEG_VER_BUILD;
  m.attr("NVJPEG_FLAGS_DEFAULT") = NVJPEG_FLAGS_DEFAULT;
  m.attr("NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE") = NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE;
  m.attr("NVJPEG_FLAGS_ENABLE_MEMORY_POOLS") = NVJPEG_FLAGS_ENABLE_MEMORY_POOLS;
  m.attr("NVJPEG_FLAGS_BITSTREAM_STRICT") = NVJPEG_FLAGS_BITSTREAM_STRICT;
  m.attr("NVJPEG_FLAGS_REDUCED_MEMORY_DECODE") = NVJPEG_FLAGS_REDUCED_MEMORY_DECODE;
  m.attr("NVJPEG_FLAGS_REDUCED_MEMORY_DECODE_ZERO_COPY") =
      NVJPEG_FLAGS_REDUCED_MEMORY_DECODE_ZERO_COPY;
  m.attr("NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION") =
      NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION;

  // -------------------------------------------------------------------------
  // Enums
  // -------------------------------------------------------------------------
  nb::enum_<nvjpegStatus_t>(m, "nvjpegStatus_t")
      .value("SUCCESS", NVJPEG_STATUS_SUCCESS)
      .value("NOT_INITIALIZED", NVJPEG_STATUS_NOT_INITIALIZED)
      .value("INVALID_PARAMETER", NVJPEG_STATUS_INVALID_PARAMETER)
      .value("BAD_JPEG", NVJPEG_STATUS_BAD_JPEG)
      .value("JPEG_NOT_SUPPORTED", NVJPEG_STATUS_JPEG_NOT_SUPPORTED)
      .value("ALLOCATOR_FAILURE", NVJPEG_STATUS_ALLOCATOR_FAILURE)
      .value("EXECUTION_FAILED", NVJPEG_STATUS_EXECUTION_FAILED)
      .value("ARCH_MISMATCH", NVJPEG_STATUS_ARCH_MISMATCH)
      .value("INTERNAL_ERROR", NVJPEG_STATUS_INTERNAL_ERROR)
      .value("IMPLEMENTATION_NOT_SUPPORTED", NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED)
      .value("INCOMPLETE_BITSTREAM", NVJPEG_STATUS_INCOMPLETE_BITSTREAM);

  nb::enum_<nvjpegExifOrientation_t>(m, "nvjpegExifOrientation_t")
      .value("UNKNOWN", NVJPEG_ORIENTATION_UNKNOWN)
      .value("NORMAL", NVJPEG_ORIENTATION_NORMAL)
      .value("FLIP_HORIZONTAL", NVJPEG_ORIENTATION_FLIP_HORIZONTAL)
      .value("ROTATE_180", NVJPEG_ORIENTATION_ROTATE_180)
      .value("FLIP_VERTICAL", NVJPEG_ORIENTATION_FLIP_VERTICAL)
      .value("TRANSPOSE", NVJPEG_ORIENTATION_TRANSPOSE)
      .value("ROTATE_90", NVJPEG_ORIENTATION_ROTATE_90)
      .value("TRANSVERSE", NVJPEG_ORIENTATION_TRANSVERSE)
      .value("ROTATE_270", NVJPEG_ORIENTATION_ROTATE_270);

  nb::enum_<nvjpegChromaSubsampling_t>(m, "nvjpegChromaSubsampling_t")
      .value("CSS_444", NVJPEG_CSS_444)
      .value("CSS_422", NVJPEG_CSS_422)
      .value("CSS_420", NVJPEG_CSS_420)
      .value("CSS_440", NVJPEG_CSS_440)
      .value("CSS_411", NVJPEG_CSS_411)
      .value("CSS_410", NVJPEG_CSS_410)
      .value("CSS_GRAY", NVJPEG_CSS_GRAY)
      .value("CSS_410V", NVJPEG_CSS_410V)
      .value("CSS_UNKNOWN", NVJPEG_CSS_UNKNOWN);

  nb::enum_<nvjpegOutputFormat_t>(m, "nvjpegOutputFormat_t")
      .value("UNCHANGED", NVJPEG_OUTPUT_UNCHANGED)
      .value("YUV", NVJPEG_OUTPUT_YUV)
      .value("Y", NVJPEG_OUTPUT_Y)
      .value("RGB", NVJPEG_OUTPUT_RGB)
      .value("BGR", NVJPEG_OUTPUT_BGR)
      .value("RGBI", NVJPEG_OUTPUT_RGBI)
      .value("BGRI", NVJPEG_OUTPUT_BGRI)
      .value("UNCHANGEDI_U16", NVJPEG_OUTPUT_UNCHANGEDI_U16)
      .value("NV12", NVJPEG_OUTPUT_NV12)
      .value("YUY2", NVJPEG_OUTPUT_YUY2);

  nb::enum_<nvjpegInputFormat_t>(m, "nvjpegInputFormat_t")
      .value("YUV", NVJPEG_INPUT_YUV)
      .value("RGB", NVJPEG_INPUT_RGB)
      .value("BGR", NVJPEG_INPUT_BGR)
      .value("RGBI", NVJPEG_INPUT_RGBI)
      .value("BGRI", NVJPEG_INPUT_BGRI)
      .value("NV12", NVJPEG_INPUT_NV12);

  nb::enum_<nvjpegBackend_t>(m, "nvjpegBackend_t")
      .value("DEFAULT", NVJPEG_BACKEND_DEFAULT)
      .value("HYBRID", NVJPEG_BACKEND_HYBRID)
      .value("GPU_HYBRID", NVJPEG_BACKEND_GPU_HYBRID)
      .value("HARDWARE", NVJPEG_BACKEND_HARDWARE)
      .value("GPU_HYBRID_DEVICE", NVJPEG_BACKEND_GPU_HYBRID_DEVICE)
      .value("HARDWARE_DEVICE", NVJPEG_BACKEND_HARDWARE_DEVICE)
      .value("LOSSLESS_JPEG", NVJPEG_BACKEND_LOSSLESS_JPEG);

  nb::enum_<nvjpegEncBackend_t>(m, "nvjpegEncBackend_t")
      .value("DEFAULT", NVJPEG_ENC_BACKEND_DEFAULT)
      .value("GPU", NVJPEG_ENC_BACKEND_GPU)
      .value("HARDWARE", NVJPEG_ENC_BACKEND_HARDWARE);

  nb::enum_<nvjpegJpegEncoding_t>(m, "nvjpegJpegEncoding_t")
      .value("UNKNOWN", NVJPEG_ENCODING_UNKNOWN)
      .value("BASELINE_DCT", NVJPEG_ENCODING_BASELINE_DCT)
      .value("EXTENDED_SEQUENTIAL_DCT_HUFFMAN",
             NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN)
      .value("PROGRESSIVE_DCT_HUFFMAN", NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)
      .value("LOSSLESS_HUFFMAN", NVJPEG_ENCODING_LOSSLESS_HUFFMAN);

  nb::enum_<nvjpegScaleFactor_t>(m, "nvjpegScaleFactor_t")
      .value("SCALE_NONE", NVJPEG_SCALE_NONE)
      .value("SCALE_1_BY_2", NVJPEG_SCALE_1_BY_2)
      .value("SCALE_1_BY_4", NVJPEG_SCALE_1_BY_4)
      .value("SCALE_1_BY_8", NVJPEG_SCALE_1_BY_8);

  nb::enum_<libraryPropertyType>(m, "libraryPropertyType")
      .value("MAJOR_VERSION", MAJOR_VERSION)
      .value("MINOR_VERSION", MINOR_VERSION)
      .value("PATCH_LEVEL", PATCH_LEVEL);

  // -------------------------------------------------------------------------
  // Types
  // -------------------------------------------------------------------------
  def_opaque<NvjpegHandle>(m, "NvjpegHandle", &NvjpegHandle::handle, "nvjpegHandle_t");
  def_opaque<NvjpegJpegState>(m, "NvjpegJpegState", &NvjpegJpegState::state,
                              "nvjpegJpegState_t (also the decoder state)");
  def_opaque<NvjpegEncoderState>(m, "NvjpegEncoderState", &NvjpegEncoderState::state,
                                 "nvjpegEncoderState_t");
  def_opaque<NvjpegEncoderParams>(m, "NvjpegEncoderParams",
                                  &NvjpegEncoderParams::params, "nvjpegEncoderParams_t");
  def_opaque<NvjpegBufferPinned>(m, "NvjpegBufferPinned", &NvjpegBufferPinned::buffer,
                                 "nvjpegBufferPinned_t");
  def_opaque<NvjpegBufferDevice>(m, "NvjpegBufferDevice", &NvjpegBufferDevice::buffer,
                                 "nvjpegBufferDevice_t");
  def_opaque<NvjpegJpegStream>(m, "NvjpegJpegStream", &NvjpegJpegStream::stream,
                               "nvjpegJpegStream_t");
  def_opaque<NvjpegDecodeParams>(m, "NvjpegDecodeParams", &NvjpegDecodeParams::params,
                                 "nvjpegDecodeParams_t");
  def_opaque<NvjpegJpegDecoder>(m, "NvjpegJpegDecoder", &NvjpegJpegDecoder::decoder,
                                "nvjpegJpegDecoder_t");

  nb::class_<NvjpegImage>(m, "NvjpegImage",
                          "nvjpegImage_t: per-channel device pointers and pitches "
                          "(bytes per row). Interleaved formats use channel 0 only.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](NvjpegImage *self, const std::vector<uintptr_t> &channel,
             const std::vector<size_t> &pitch) {
            if (channel.size() > NVJPEG_MAX_COMPONENT || pitch.size() != channel.size())
              throw nb::value_error(
                  "channel and pitch must have equal length <= NVJPEG_MAX_COMPONENT");
            new (self) NvjpegImage();
            for (size_t i = 0; i < channel.size(); ++i) {
              self->image.channel[i] = as_bytes(channel[i]);
              self->image.pitch[i] = pitch[i];
            }
          },
          "channel"_a, "pitch"_a)
      .def_prop_rw(
          "channel",
          [](const NvjpegImage &img) {
            std::vector<uintptr_t> v(NVJPEG_MAX_COMPONENT);
            for (int i = 0; i < NVJPEG_MAX_COMPONENT; ++i)
              v[i] = reinterpret_cast<uintptr_t>(img.image.channel[i]);
            return v;
          },
          [](NvjpegImage &img, const std::vector<uintptr_t> &v) {
            if (v.size() > NVJPEG_MAX_COMPONENT)
              throw nb::value_error("at most NVJPEG_MAX_COMPONENT channels");
            for (int i = 0; i < NVJPEG_MAX_COMPONENT; ++i)
              img.image.channel[i] = i < (int)v.size() ? as_bytes(v[i]) : nullptr;
          },
          "Device pointers, one per component (unused ones are 0).")
      .def_prop_rw(
          "pitch",
          [](const NvjpegImage &img) {
            return std::vector<size_t>(img.image.pitch,
                                       img.image.pitch + NVJPEG_MAX_COMPONENT);
          },
          [](NvjpegImage &img, const std::vector<size_t> &v) {
            if (v.size() > NVJPEG_MAX_COMPONENT)
              throw nb::value_error("at most NVJPEG_MAX_COMPONENT channels");
            for (int i = 0; i < NVJPEG_MAX_COMPONENT; ++i)
              img.image.pitch[i] = i < (int)v.size() ? v[i] : 0;
          },
          "Row pitch in bytes, one per component.")
      .def("__repr__", [](const NvjpegImage &img) {
        std::string s = "<pynvjpeg.NvjpegImage";
        for (int i = 0; i < NVJPEG_MAX_COMPONENT; ++i) {
          char buf[64];
          std::snprintf(buf, sizeof(buf), " [0x%zx, %zu]",
                        static_cast<size_t>(reinterpret_cast<uintptr_t>(img.image.channel[i])),
                        img.image.pitch[i]);
          s += buf;
        }
        return s + ">";
      });

  nb::class_<NvjpegDevAllocator>(m, "NvjpegDevAllocator",
                                 "nvjpegDevAllocator_t. Fields are C function "
                                 "pointer addresses: int malloc(void**, size_t), "
                                 "int free(void*).")
      .def(nb::init<>())
      .def(
          "__init__",
          [](NvjpegDevAllocator *self, uintptr_t a, uintptr_t f) {
            new (self) NvjpegDevAllocator{a, f};
          },
          "dev_malloc"_a, "dev_free"_a)
      .def_rw("dev_malloc", &NvjpegDevAllocator::dev_malloc)
      .def_rw("dev_free", &NvjpegDevAllocator::dev_free);

  nb::class_<NvjpegPinnedAllocator>(m, "NvjpegPinnedAllocator",
                                    "nvjpegPinnedAllocator_t. Fields are C function "
                                    "pointer addresses: int malloc(void**, size_t, "
                                    "unsigned int flags), int free(void*).")
      .def(nb::init<>())
      .def(
          "__init__",
          [](NvjpegPinnedAllocator *self, uintptr_t a, uintptr_t f) {
            new (self) NvjpegPinnedAllocator{a, f};
          },
          "pinned_malloc"_a, "pinned_free"_a)
      .def_rw("pinned_malloc", &NvjpegPinnedAllocator::pinned_malloc)
      .def_rw("pinned_free", &NvjpegPinnedAllocator::pinned_free);

  nb::class_<NvjpegDevAllocatorV2>(m, "NvjpegDevAllocatorV2",
                                   "nvjpegDevAllocatorV2_t. Function pointers: "
                                   "int malloc(void* ctx, void**, size_t, cudaStream_t), "
                                   "int free(void* ctx, void*, size_t, cudaStream_t).")
      .def(nb::init<>())
      .def(
          "__init__",
          [](NvjpegDevAllocatorV2 *self, uintptr_t a, uintptr_t f, uintptr_t c) {
            new (self) NvjpegDevAllocatorV2{a, f, c};
          },
          "dev_malloc"_a, "dev_free"_a, "dev_ctx"_a = 0)
      .def_rw("dev_malloc", &NvjpegDevAllocatorV2::dev_malloc)
      .def_rw("dev_free", &NvjpegDevAllocatorV2::dev_free)
      .def_rw("dev_ctx", &NvjpegDevAllocatorV2::dev_ctx);

  nb::class_<NvjpegPinnedAllocatorV2>(m, "NvjpegPinnedAllocatorV2",
                                      "nvjpegPinnedAllocatorV2_t. Function pointers: "
                                      "int malloc(void* ctx, void**, size_t, cudaStream_t), "
                                      "int free(void* ctx, void*, size_t, cudaStream_t).")
      .def(nb::init<>())
      .def(
          "__init__",
          [](NvjpegPinnedAllocatorV2 *self, uintptr_t a, uintptr_t f, uintptr_t c) {
            new (self) NvjpegPinnedAllocatorV2{a, f, c};
          },
          "pinned_malloc"_a, "pinned_free"_a, "pinned_ctx"_a = 0)
      .def_rw("pinned_malloc", &NvjpegPinnedAllocatorV2::pinned_malloc)
      .def_rw("pinned_free", &NvjpegPinnedAllocatorV2::pinned_free)
      .def_rw("pinned_ctx", &NvjpegPinnedAllocatorV2::pinned_ctx);

  // NvjpegImage helpers kept from the original API.
  m.def(
      "nvjpeg_image_set_channel",
      [](NvjpegImage &image, int channel_idx, uintptr_t ptr, size_t pitch) {
        check_channel(channel_idx);
        image.image.channel[channel_idx] = as_bytes(ptr);
        image.image.pitch[channel_idx] = pitch;
      },
      "image"_a, "channel_idx"_a, "ptr"_a, "pitch"_a);
  m.def(
      "nvjpeg_image_get_channel_ptr",
      [](const NvjpegImage &image, int channel_idx) {
        check_channel(channel_idx);
        return reinterpret_cast<uintptr_t>(image.image.channel[channel_idx]);
      },
      "image"_a, "channel_idx"_a);
  m.def(
      "nvjpeg_image_get_pitch",
      [](const NvjpegImage &image, int channel_idx) {
        check_channel(channel_idx);
        return image.image.pitch[channel_idx];
      },
      "image"_a, "channel_idx"_a);

  // -------------------------------------------------------------------------
  // Library / handle
  // -------------------------------------------------------------------------
  m.def(
      "nvjpeg_get_property",
      [](libraryPropertyType type) {
        int value = 0;
        nvjpegStatus_t s = nvjpegGetProperty(type, &value);
        return std::make_tuple(s, value);
      },
      "type"_a, Release());
  m.def(
      "nvjpeg_get_cudart_property",
      [](libraryPropertyType type) {
        int value = 0;
        nvjpegStatus_t s = nvjpegGetCudartProperty(type, &value);
        return std::make_tuple(s, value);
      },
      "type"_a, Release());

  m.def(
      "nvjpeg_create",
      [](nvjpegBackend_t backend, const NvjpegDevAllocator *dev_allocator) {
        OptAlloc<NvjpegDevAllocator, nvjpegDevAllocator_t> da(dev_allocator);
        NvjpegHandle h;
        nvjpegStatus_t s = nvjpegCreate(backend, da.ptr, &h.handle);
        return std::make_tuple(s, h);
      },
      "backend"_a = NVJPEG_BACKEND_DEFAULT, "dev_allocator"_a.none() = nb::none(),
      Release());
  m.def(
      "nvjpeg_create_simple",
      []() {
        NvjpegHandle h;
        nvjpegStatus_t s = nvjpegCreateSimple(&h.handle);
        return std::make_tuple(s, h);
      },
      Release());
  m.def(
      "nvjpeg_create_ex",
      [](nvjpegBackend_t backend, const NvjpegDevAllocator *dev_allocator,
         const NvjpegPinnedAllocator *pinned_allocator, unsigned int flags) {
        OptAlloc<NvjpegDevAllocator, nvjpegDevAllocator_t> da(dev_allocator);
        OptAlloc<NvjpegPinnedAllocator, nvjpegPinnedAllocator_t> pa(pinned_allocator);
        NvjpegHandle h;
        nvjpegStatus_t s = nvjpegCreateEx(backend, da.ptr, pa.ptr, flags, &h.handle);
        return std::make_tuple(s, h);
      },
      "backend"_a = NVJPEG_BACKEND_DEFAULT, "dev_allocator"_a.none() = nb::none(),
      "pinned_allocator"_a.none() = nb::none(), "flags"_a = NVJPEG_FLAGS_DEFAULT,
      Release());
  m.def(
      "nvjpeg_create_ex_v2",
      [](nvjpegBackend_t backend, const NvjpegDevAllocatorV2 *dev_allocator,
         const NvjpegPinnedAllocatorV2 *pinned_allocator, unsigned int flags) {
        OptAlloc<NvjpegDevAllocatorV2, nvjpegDevAllocatorV2_t> da(dev_allocator);
        OptAlloc<NvjpegPinnedAllocatorV2, nvjpegPinnedAllocatorV2_t> pa(pinned_allocator);
        NvjpegHandle h;
        nvjpegStatus_t s = nvjpegCreateExV2(backend, da.ptr, pa.ptr, flags, &h.handle);
        return std::make_tuple(s, h);
      },
      "backend"_a = NVJPEG_BACKEND_DEFAULT, "dev_allocator"_a.none() = nb::none(),
      "pinned_allocator"_a.none() = nb::none(), "flags"_a = NVJPEG_FLAGS_DEFAULT,
      Release());
  m.def(
      "nvjpeg_destroy",
      [](NvjpegHandle &handle) {
        nvjpegStatus_t s = nvjpegDestroy(handle.handle);
        if (s == NVJPEG_STATUS_SUCCESS) handle.handle = nullptr;
        return s;
      },
      "handle"_a, Release());

  m.def(
      "nvjpeg_set_device_memory_padding",
      [](size_t padding, const NvjpegHandle &handle) {
        return nvjpegSetDeviceMemoryPadding(padding, handle.handle);
      },
      "padding"_a, "handle"_a, Release());
  m.def(
      "nvjpeg_get_device_memory_padding",
      [](const NvjpegHandle &handle) {
        size_t padding = 0;
        nvjpegStatus_t s = nvjpegGetDeviceMemoryPadding(&padding, handle.handle);
        return std::make_tuple(s, padding);
      },
      "handle"_a, Release());
  m.def(
      "nvjpeg_set_pinned_memory_padding",
      [](size_t padding, const NvjpegHandle &handle) {
        return nvjpegSetPinnedMemoryPadding(padding, handle.handle);
      },
      "padding"_a, "handle"_a, Release());
  m.def(
      "nvjpeg_get_pinned_memory_padding",
      [](const NvjpegHandle &handle) {
        size_t padding = 0;
        nvjpegStatus_t s = nvjpegGetPinnedMemoryPadding(&padding, handle.handle);
        return std::make_tuple(s, padding);
      },
      "handle"_a, Release());

  m.def(
      "nvjpeg_get_hardware_decoder_info",
      [](const NvjpegHandle &handle) {
        unsigned int num_engines = 0, num_cores_per_engine = 0;
        nvjpegStatus_t s = nvjpegGetHardwareDecoderInfo(handle.handle, &num_engines,
                                                        &num_cores_per_engine);
        return std::make_tuple(s, num_engines, num_cores_per_engine);
      },
      "handle"_a, Release());
  m.def(
      "nvjpeg_get_hardware_encoder_info",
      [](const NvjpegHandle &handle) {
        unsigned int num_engines = 0;
        nvjpegStatus_t s = nvjpegGetHardwareEncoderInfo(handle.handle, &num_engines);
        return std::make_tuple(s, num_engines);
      },
      "handle"_a, Release());

  // -------------------------------------------------------------------------
  // Legacy decode API (nvjpegJpegState + nvjpegDecode / batched)
  // -------------------------------------------------------------------------
  m.def(
      "nvjpeg_jpeg_state_create",
      [](const NvjpegHandle &handle) {
        NvjpegJpegState st;
        nvjpegStatus_t s = nvjpegJpegStateCreate(handle.handle, &st.state);
        return std::make_tuple(s, st);
      },
      "handle"_a, Release());
  m.def(
      "nvjpeg_jpeg_state_destroy",
      [](NvjpegJpegState &state) {
        nvjpegStatus_t s = nvjpegJpegStateDestroy(state.state);
        if (s == NVJPEG_STATUS_SUCCESS) state.state = nullptr;
        return s;
      },
      "jpeg_handle"_a, Release());

  m.def(
      "nvjpeg_get_image_info",
      [](const NvjpegHandle &handle, nb::handle data, std::optional<size_t> length) {
        HostData d(data, length);
        int n_components = 0;
        nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;
        std::vector<int> widths(NVJPEG_MAX_COMPONENT), heights(NVJPEG_MAX_COMPONENT);
        nvjpegStatus_t s;
        {
          nb::gil_scoped_release r;
          s = nvjpegGetImageInfo(handle.handle, d.ptr(), d.len(), &n_components,
                                 &subsampling, widths.data(), heights.data());
        }
        return std::make_tuple(s, n_components, subsampling, widths, heights);
      },
      "handle"_a, "data"_a, "length"_a.none() = nb::none(),
      "Returns (status, n_components, subsampling, widths[4], heights[4]).");

  m.def(
      "nvjpeg_decode",
      [](const NvjpegHandle &handle, const NvjpegJpegState &jpeg_handle, nb::handle data,
         std::optional<size_t> length, nvjpegOutputFormat_t output_format,
         NvjpegImage &destination, uintptr_t stream) {
        HostData d(data, length);
        nb::gil_scoped_release r;
        return nvjpegDecode(handle.handle, jpeg_handle.state, d.ptr(), d.len(),
                            output_format, &destination.image, as_stream(stream));
      },
      "handle"_a, "jpeg_handle"_a, "data"_a, "length"_a.none(), "output_format"_a,
      "destination"_a, "stream"_a = 0,
      "Pass length=None to use the full size of a buffer-protocol `data`.");

  m.def(
      "nvjpeg_decode_batched_initialize",
      [](const NvjpegHandle &handle, const NvjpegJpegState &jpeg_handle, int batch_size,
         int max_cpu_threads, nvjpegOutputFormat_t output_format) {
        return nvjpegDecodeBatchedInitialize(handle.handle, jpeg_handle.state, batch_size,
                                             max_cpu_threads, output_format);
      },
      "handle"_a, "jpeg_handle"_a, "batch_size"_a, "max_cpu_threads"_a, "output_format"_a,
      Release());

  m.def(
      "nvjpeg_decode_batched",
      [](const NvjpegHandle &handle, const NvjpegJpegState &jpeg_handle,
         nb::sequence data, std::optional<std::vector<size_t>> lengths,
         const std::vector<NvjpegImage *> &destinations, uintptr_t stream) {
        auto items = host_data_list(data, lengths);
        if (destinations.size() != items.size())
          throw nb::value_error("`destinations` must have the same length as `data`");
        std::vector<const unsigned char *> ptrs;
        std::vector<size_t> lens;
        std::vector<nvjpegImage_t> dst;
        for (size_t i = 0; i < items.size(); ++i) {
          ptrs.push_back(items[i].ptr());
          lens.push_back(items[i].len());
          dst.push_back(destinations[i]->image);
        }
        nb::gil_scoped_release r;
        return nvjpegDecodeBatched(handle.handle, jpeg_handle.state, ptrs.data(),
                                   lens.data(), dst.data(), as_stream(stream));
      },
      "handle"_a, "jpeg_handle"_a, "data"_a, "lengths"_a.none(), "destinations"_a,
      "stream"_a = 0,
      "`data` is a sequence of bitstreams (buffers or pointers); `lengths` may be "
      "None when every item is a buffer.");

  m.def(
      "nvjpeg_decode_batched_pre_allocate",
      [](const NvjpegHandle &handle, const NvjpegJpegState &jpeg_handle, int batch_size,
         int width, int height, nvjpegChromaSubsampling_t chroma_subsampling,
         nvjpegOutputFormat_t output_format) {
        return nvjpegDecodeBatchedPreAllocate(handle.handle, jpeg_handle.state,
                                              batch_size, width, height,
                                              chroma_subsampling, output_format);
      },
      "handle"_a, "jpeg_handle"_a, "batch_size"_a, "width"_a, "height"_a,
      "chroma_subsampling"_a, "output_format"_a, Release());

  m.def(
      "nvjpeg_decode_batched_parse_jpeg_tables",
      [](const NvjpegHandle &handle, const NvjpegJpegState &jpeg_handle, nb::handle data,
         std::optional<size_t> length) {
        HostData d(data, length);
        nb::gil_scoped_release r;
        return nvjpegDecodeBatchedParseJpegTables(handle.handle, jpeg_handle.state,
                                                  d.ptr(), d.len());
      },
      "handle"_a, "jpeg_handle"_a, "data"_a, "length"_a.none() = nb::none());

  // -------------------------------------------------------------------------
  // Encoder
  // -------------------------------------------------------------------------
  m.def(
      "nvjpeg_encoder_state_create",
      [](const NvjpegHandle &handle, uintptr_t stream) {
        NvjpegEncoderState st;
        nvjpegStatus_t s = nvjpegEncoderStateCreate(handle.handle, &st.state,
                                                    as_stream(stream));
        return std::make_tuple(s, st);
      },
      "handle"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_encoder_state_create_with_backend",
      [](const NvjpegHandle &handle, nvjpegEncBackend_t backend, uintptr_t stream) {
        NvjpegEncoderState st;
        nvjpegStatus_t s = nvjpegEncoderStateCreateWithBackend(
            handle.handle, &st.state, backend, as_stream(stream));
        return std::make_tuple(s, st);
      },
      "handle"_a, "backend"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_encoder_state_destroy",
      [](NvjpegEncoderState &state) {
        nvjpegStatus_t s = nvjpegEncoderStateDestroy(state.state);
        if (s == NVJPEG_STATUS_SUCCESS) state.state = nullptr;
        return s;
      },
      "encoder_state"_a, Release());

  m.def(
      "nvjpeg_encoder_params_create",
      [](const NvjpegHandle &handle, uintptr_t stream) {
        NvjpegEncoderParams p;
        nvjpegStatus_t s =
            nvjpegEncoderParamsCreate(handle.handle, &p.params, as_stream(stream));
        return std::make_tuple(s, p);
      },
      "handle"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_encoder_params_destroy",
      [](NvjpegEncoderParams &params) {
        nvjpegStatus_t s = nvjpegEncoderParamsDestroy(params.params);
        if (s == NVJPEG_STATUS_SUCCESS) params.params = nullptr;
        return s;
      },
      "encoder_params"_a, Release());

  m.def(
      "nvjpeg_encoder_params_set_quality",
      [](const NvjpegEncoderParams &params, int quality, uintptr_t stream) {
        return nvjpegEncoderParamsSetQuality(params.params, quality, as_stream(stream));
      },
      "encoder_params"_a, "quality"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_encoder_params_set_encoding",
      [](const NvjpegEncoderParams &params, nvjpegJpegEncoding_t etype, uintptr_t stream) {
        return nvjpegEncoderParamsSetEncoding(params.params, etype, as_stream(stream));
      },
      "encoder_params"_a, "etype"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_encoder_params_set_optimized_huffman",
      [](const NvjpegEncoderParams &params, int optimized, uintptr_t stream) {
        return nvjpegEncoderParamsSetOptimizedHuffman(params.params, optimized,
                                                      as_stream(stream));
      },
      "encoder_params"_a, "optimized"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_encoder_params_set_sampling_factors",
      [](const NvjpegEncoderParams &params, nvjpegChromaSubsampling_t chroma_subsampling,
         uintptr_t stream) {
        return nvjpegEncoderParamsSetSamplingFactors(params.params, chroma_subsampling,
                                                     as_stream(stream));
      },
      "encoder_params"_a, "chroma_subsampling"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_encoder_params_set_restart_interval",
      [](const NvjpegEncoderParams &params, unsigned int restart_interval,
         uintptr_t stream) {
        return nvjpegEncoderParamsSetRestartInterval(params.params, restart_interval,
                                                     as_stream(stream));
      },
      "encoder_params"_a, "restart_interval"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_encoder_params_copy_metadata",
      [](const NvjpegEncoderState &state, const NvjpegEncoderParams &params,
         const NvjpegJpegStream &jpeg_stream, uintptr_t stream) {
        return nvjpegEncoderParamsCopyMetadata(state.state, params.params,
                                               jpeg_stream.stream, as_stream(stream));
      },
      "encoder_state"_a, "encode_params"_a, "jpeg_stream"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_encoder_params_copy_quantization_tables",
      [](const NvjpegEncoderParams &params, const NvjpegJpegStream &jpeg_stream,
         uintptr_t stream) {
        return nvjpegEncoderParamsCopyQuantizationTables(params.params, jpeg_stream.stream,
                                                         as_stream(stream));
      },
      "encode_params"_a, "jpeg_stream"_a, "stream"_a = 0, Release());

  m.def(
      "nvjpeg_encode_get_buffer_size",
      [](const NvjpegHandle &handle, const NvjpegEncoderParams &params, int image_width,
         int image_height) {
        size_t max_stream_length = 0;
        nvjpegStatus_t s = nvjpegEncodeGetBufferSize(handle.handle, params.params,
                                                     image_width, image_height,
                                                     &max_stream_length);
        return std::make_tuple(s, max_stream_length);
      },
      "handle"_a, "encoder_params"_a, "image_width"_a, "image_height"_a, Release());

  m.def(
      "nvjpeg_encode_yuv",
      [](const NvjpegHandle &handle, const NvjpegEncoderState &state,
         const NvjpegEncoderParams &params, const NvjpegImage &source,
         nvjpegChromaSubsampling_t chroma_subsampling, int image_width, int image_height,
         uintptr_t stream) {
        return nvjpegEncodeYUV(handle.handle, state.state, params.params, &source.image,
                               chroma_subsampling, image_width, image_height,
                               as_stream(stream));
      },
      "handle"_a, "encoder_state"_a, "encoder_params"_a, "source"_a,
      "chroma_subsampling"_a, "image_width"_a, "image_height"_a, "stream"_a = 0,
      Release());
  m.def(
      "nvjpeg_encode_image",
      [](const NvjpegHandle &handle, const NvjpegEncoderState &state,
         const NvjpegEncoderParams &params, const NvjpegImage &source,
         nvjpegInputFormat_t input_format, int image_width, int image_height,
         uintptr_t stream) {
        return nvjpegEncodeImage(handle.handle, state.state, params.params, &source.image,
                                 input_format, image_width, image_height,
                                 as_stream(stream));
      },
      "handle"_a, "encoder_state"_a, "encoder_params"_a, "source"_a, "input_format"_a,
      "image_width"_a, "image_height"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_encode",
      [](const NvjpegHandle &handle, const NvjpegEncoderState &state,
         const NvjpegEncoderParams &params, const NvjpegImage &source,
         nvjpegChromaSubsampling_t input_subsampling, nvjpegInputFormat_t input_format,
         int image_width, int image_height, uintptr_t stream) {
        return nvjpegEncode(handle.handle, state.state, params.params, &source.image,
                            input_subsampling, input_format, image_width, image_height,
                            as_stream(stream));
      },
      "handle"_a, "encoder_state"_a, "encoder_params"_a, "source"_a,
      "input_subsampling"_a, "input_format"_a, "image_width"_a, "image_height"_a,
      "stream"_a = 0, Release());

  m.def(
      "nvjpeg_encode_retrieve_bitstream",
      [](const NvjpegHandle &handle, const NvjpegEncoderState &state, uintptr_t data,
         size_t length, uintptr_t stream) {
        nvjpegStatus_t s = nvjpegEncodeRetrieveBitstream(
            handle.handle, state.state, as_bytes(data), &length, as_stream(stream));
        return std::make_tuple(s, length);
      },
      "handle"_a, "encoder_state"_a, "data"_a, "length"_a, "stream"_a = 0, Release(),
      "Copies the bitstream into host memory at `data` (capacity `length` bytes). "
      "Returns (status, length) with the actual bitstream size. data=0 queries "
      "the size only.");
  m.def(
      "nvjpeg_encode_retrieve_bitstream_device",
      [](const NvjpegHandle &handle, const NvjpegEncoderState &state, uintptr_t data,
         size_t length, uintptr_t stream) {
        nvjpegStatus_t s = nvjpegEncodeRetrieveBitstreamDevice(
            handle.handle, state.state, as_bytes(data), &length, as_stream(stream));
        return std::make_tuple(s, length);
      },
      "handle"_a, "encoder_state"_a, "data"_a, "length"_a, "stream"_a = 0, Release(),
      "Like nvjpeg_encode_retrieve_bitstream, but `data` is device memory.");

  // Conveniences from the original API (thin aliases of the call above).
  m.def(
      "nvjpeg_encode_retrieve_bitstream_size",
      [](const NvjpegHandle &handle, const NvjpegEncoderState &state, uintptr_t stream) {
        size_t length = 0;
        nvjpegStatus_t s = nvjpegEncodeRetrieveBitstream(handle.handle, state.state,
                                                         nullptr, &length,
                                                         as_stream(stream));
        return std::make_tuple(s, length);
      },
      "handle"_a, "encoder_state"_a, "stream"_a = 0, Release(),
      "nvjpegEncodeRetrieveBitstream with data=NULL: returns (status, size).");
  m.def(
      "nvjpeg_encode_retrieve_bitstream_with_size",
      [](const NvjpegHandle &handle, const NvjpegEncoderState &state, uintptr_t data_ptr,
         size_t buffer_size, uintptr_t stream) {
        size_t length = buffer_size;
        nvjpegStatus_t s = nvjpegEncodeRetrieveBitstream(
            handle.handle, state.state, as_bytes(data_ptr), &length, as_stream(stream));
        return std::make_tuple(s, length);
      },
      "handle"_a, "encoder_state"_a, "data_ptr"_a, "buffer_size"_a, "stream"_a = 0,
      Release(), "Alias of nvjpeg_encode_retrieve_bitstream.");

  // -------------------------------------------------------------------------
  // Buffers
  // -------------------------------------------------------------------------
  m.def(
      "nvjpeg_buffer_pinned_create",
      [](const NvjpegHandle &handle, const NvjpegPinnedAllocator *pinned_allocator) {
        OptAlloc<NvjpegPinnedAllocator, nvjpegPinnedAllocator_t> pa(pinned_allocator);
        NvjpegBufferPinned b;
        nvjpegStatus_t s = nvjpegBufferPinnedCreate(handle.handle, pa.ptr, &b.buffer);
        return std::make_tuple(s, b);
      },
      "handle"_a, "pinned_allocator"_a.none() = nb::none(), Release());
  m.def(
      "nvjpeg_buffer_pinned_create_v2",
      [](const NvjpegHandle &handle, const NvjpegPinnedAllocatorV2 *pinned_allocator) {
        OptAlloc<NvjpegPinnedAllocatorV2, nvjpegPinnedAllocatorV2_t> pa(pinned_allocator);
        NvjpegBufferPinned b;
        nvjpegStatus_t s = nvjpegBufferPinnedCreateV2(handle.handle, pa.ptr, &b.buffer);
        return std::make_tuple(s, b);
      },
      "handle"_a, "pinned_allocator"_a, Release());
  m.def(
      "nvjpeg_buffer_pinned_resize",
      [](const NvjpegBufferPinned &buffer, size_t size, uintptr_t stream) {
        return nvjpegBufferPinnedResize(buffer.buffer, size, as_stream(stream));
      },
      "buffer"_a, "size"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_buffer_pinned_destroy",
      [](NvjpegBufferPinned &buffer) {
        nvjpegStatus_t s = nvjpegBufferPinnedDestroy(buffer.buffer);
        if (s == NVJPEG_STATUS_SUCCESS) buffer.buffer = nullptr;
        return s;
      },
      "buffer"_a, Release());
  m.def(
      "nvjpeg_buffer_pinned_retrieve",
      [](const NvjpegBufferPinned &buffer) {
        size_t size = 0;
        void *ptr = nullptr;
        nvjpegStatus_t s = nvjpegBufferPinnedRetrieve(buffer.buffer, &size, &ptr);
        return std::make_tuple(s, size, reinterpret_cast<uintptr_t>(ptr));
      },
      "buffer"_a, Release(), "Returns (status, size, ptr).");

  m.def(
      "nvjpeg_buffer_device_create",
      [](const NvjpegHandle &handle, const NvjpegDevAllocator *device_allocator) {
        OptAlloc<NvjpegDevAllocator, nvjpegDevAllocator_t> da(device_allocator);
        NvjpegBufferDevice b;
        nvjpegStatus_t s = nvjpegBufferDeviceCreate(handle.handle, da.ptr, &b.buffer);
        return std::make_tuple(s, b);
      },
      "handle"_a, "device_allocator"_a.none() = nb::none(), Release());
  m.def(
      "nvjpeg_buffer_device_create_v2",
      [](const NvjpegHandle &handle, const NvjpegDevAllocatorV2 *device_allocator) {
        OptAlloc<NvjpegDevAllocatorV2, nvjpegDevAllocatorV2_t> da(device_allocator);
        NvjpegBufferDevice b;
        nvjpegStatus_t s = nvjpegBufferDeviceCreateV2(handle.handle, da.ptr, &b.buffer);
        return std::make_tuple(s, b);
      },
      "handle"_a, "device_allocator"_a, Release());
  m.def(
      "nvjpeg_buffer_device_resize",
      [](const NvjpegBufferDevice &buffer, size_t size, uintptr_t stream) {
        return nvjpegBufferDeviceResize(buffer.buffer, size, as_stream(stream));
      },
      "buffer"_a, "size"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_buffer_device_destroy",
      [](NvjpegBufferDevice &buffer) {
        nvjpegStatus_t s = nvjpegBufferDeviceDestroy(buffer.buffer);
        if (s == NVJPEG_STATUS_SUCCESS) buffer.buffer = nullptr;
        return s;
      },
      "buffer"_a, Release());
  m.def(
      "nvjpeg_buffer_device_retrieve",
      [](const NvjpegBufferDevice &buffer) {
        size_t size = 0;
        void *ptr = nullptr;
        nvjpegStatus_t s = nvjpegBufferDeviceRetrieve(buffer.buffer, &size, &ptr);
        return std::make_tuple(s, size, reinterpret_cast<uintptr_t>(ptr));
      },
      "buffer"_a, Release(), "Returns (status, size, ptr).");

  m.def(
      "nvjpeg_state_attach_pinned_buffer",
      [](const NvjpegJpegState &decoder_state, const NvjpegBufferPinned &pinned_buffer) {
        return nvjpegStateAttachPinnedBuffer(decoder_state.state, pinned_buffer.buffer);
      },
      "decoder_state"_a, "pinned_buffer"_a, Release());
  m.def(
      "nvjpeg_state_attach_device_buffer",
      [](const NvjpegJpegState &decoder_state, const NvjpegBufferDevice &device_buffer) {
        return nvjpegStateAttachDeviceBuffer(decoder_state.state, device_buffer.buffer);
      },
      "decoder_state"_a, "device_buffer"_a, Release());

  // -------------------------------------------------------------------------
  // JPEG stream (parsing)
  // -------------------------------------------------------------------------
  m.def(
      "nvjpeg_jpeg_stream_create",
      [](const NvjpegHandle &handle) {
        NvjpegJpegStream js;
        nvjpegStatus_t s = nvjpegJpegStreamCreate(handle.handle, &js.stream);
        return std::make_tuple(s, js);
      },
      "handle"_a, Release());
  m.def(
      "nvjpeg_jpeg_stream_destroy",
      [](NvjpegJpegStream &jpeg_stream) {
        nvjpegStatus_t s = nvjpegJpegStreamDestroy(jpeg_stream.stream);
        if (s == NVJPEG_STATUS_SUCCESS) jpeg_stream.stream = nullptr;
        return s;
      },
      "jpeg_stream"_a, Release());
  m.def(
      "nvjpeg_jpeg_stream_parse",
      [](const NvjpegHandle &handle, nb::handle data, std::optional<size_t> length,
         int save_metadata, int save_stream, const NvjpegJpegStream &jpeg_stream) {
        HostData d(data, length);
        nb::gil_scoped_release r;
        return nvjpegJpegStreamParse(handle.handle, d.ptr(), d.len(), save_metadata,
                                     save_stream, jpeg_stream.stream);
      },
      "handle"_a, "data"_a, "length"_a.none(), "save_metadata"_a, "save_stream"_a,
      "jpeg_stream"_a,
      "With save_stream=0 nvJPEG keeps a pointer to `data` instead of copying it: "
      "keep `data` alive and unmodified until decoding of this stream is done.");
  m.def(
      "nvjpeg_jpeg_stream_parse_header",
      [](const NvjpegHandle &handle, nb::handle data, std::optional<size_t> length,
         const NvjpegJpegStream &jpeg_stream) {
        HostData d(data, length);
        nb::gil_scoped_release r;
        return nvjpegJpegStreamParseHeader(handle.handle, d.ptr(), d.len(),
                                           jpeg_stream.stream);
      },
      "handle"_a, "data"_a, "length"_a.none(), "jpeg_stream"_a);
  m.def(
      "nvjpeg_jpeg_stream_parse_tables",
      [](const NvjpegHandle &handle, nb::handle data, std::optional<size_t> length,
         const NvjpegJpegStream &jpeg_stream) {
        HostData d(data, length);
        nb::gil_scoped_release r;
        return nvjpegJpegStreamParseTables(handle.handle, d.ptr(), d.len(),
                                           jpeg_stream.stream);
      },
      "handle"_a, "data"_a, "length"_a.none(), "jpeg_stream"_a);

  m.def(
      "nvjpeg_jpeg_stream_get_jpeg_encoding",
      [](const NvjpegJpegStream &js) {
        nvjpegJpegEncoding_t e = NVJPEG_ENCODING_UNKNOWN;
        nvjpegStatus_t s = nvjpegJpegStreamGetJpegEncoding(js.stream, &e);
        return std::make_tuple(s, e);
      },
      "jpeg_stream"_a, Release());
  m.def(
      "nvjpeg_jpeg_stream_get_frame_dimensions",
      [](const NvjpegJpegStream &js) {
        unsigned int w = 0, h = 0;
        nvjpegStatus_t s = nvjpegJpegStreamGetFrameDimensions(js.stream, &w, &h);
        return std::make_tuple(s, w, h);
      },
      "jpeg_stream"_a, Release(), "Returns (status, width, height).");
  m.def(
      "nvjpeg_jpeg_stream_get_components_num",
      [](const NvjpegJpegStream &js) {
        unsigned int n = 0;
        nvjpegStatus_t s = nvjpegJpegStreamGetComponentsNum(js.stream, &n);
        return std::make_tuple(s, n);
      },
      "jpeg_stream"_a, Release());
  m.def(
      "nvjpeg_jpeg_stream_get_component_dimensions",
      [](const NvjpegJpegStream &js, unsigned int component) {
        unsigned int w = 0, h = 0;
        nvjpegStatus_t s = nvjpegJpegStreamGetComponentDimensions(js.stream, component,
                                                                  &w, &h);
        return std::make_tuple(s, w, h);
      },
      "jpeg_stream"_a, "component"_a, Release(), "Returns (status, width, height).");
  m.def(
      "nvjpeg_jpeg_stream_get_exif_orientation",
      [](const NvjpegJpegStream &js) {
        nvjpegExifOrientation_t o = NVJPEG_ORIENTATION_UNKNOWN;
        nvjpegStatus_t s = nvjpegJpegStreamGetExifOrientation(js.stream, &o);
        return std::make_tuple(s, o);
      },
      "jpeg_stream"_a, Release());
  m.def(
      "nvjpeg_jpeg_stream_get_sample_precision",
      [](const NvjpegJpegStream &js) {
        unsigned int p = 0;
        nvjpegStatus_t s = nvjpegJpegStreamGetSamplePrecision(js.stream, &p);
        return std::make_tuple(s, p);
      },
      "jpeg_stream"_a, Release());
  m.def(
      "nvjpeg_jpeg_stream_get_chroma_subsampling",
      [](const NvjpegJpegStream &js) {
        nvjpegChromaSubsampling_t c = NVJPEG_CSS_UNKNOWN;
        nvjpegStatus_t s = nvjpegJpegStreamGetChromaSubsampling(js.stream, &c);
        return std::make_tuple(s, c);
      },
      "jpeg_stream"_a, Release());

  // -------------------------------------------------------------------------
  // Decode params
  // -------------------------------------------------------------------------
  m.def(
      "nvjpeg_decode_params_create",
      [](const NvjpegHandle &handle) {
        NvjpegDecodeParams p;
        nvjpegStatus_t s = nvjpegDecodeParamsCreate(handle.handle, &p.params);
        return std::make_tuple(s, p);
      },
      "handle"_a, Release());
  m.def(
      "nvjpeg_decode_params_destroy",
      [](NvjpegDecodeParams &p) {
        nvjpegStatus_t s = nvjpegDecodeParamsDestroy(p.params);
        if (s == NVJPEG_STATUS_SUCCESS) p.params = nullptr;
        return s;
      },
      "decode_params"_a, Release());
  m.def(
      "nvjpeg_decode_params_set_output_format",
      [](const NvjpegDecodeParams &p, nvjpegOutputFormat_t f) {
        return nvjpegDecodeParamsSetOutputFormat(p.params, f);
      },
      "decode_params"_a, "output_format"_a, Release());
  m.def(
      "nvjpeg_decode_params_set_roi",
      [](const NvjpegDecodeParams &p, int offset_x, int offset_y, int roi_width,
         int roi_height) {
        return nvjpegDecodeParamsSetROI(p.params, offset_x, offset_y, roi_width,
                                        roi_height);
      },
      "decode_params"_a, "offset_x"_a, "offset_y"_a, "roi_width"_a, "roi_height"_a,
      Release());
  m.def(
      "nvjpeg_decode_params_set_allow_cmyk",
      [](const NvjpegDecodeParams &p, int allow_cmyk) {
        return nvjpegDecodeParamsSetAllowCMYK(p.params, allow_cmyk);
      },
      "decode_params"_a, "allow_cmyk"_a, Release());
  m.def(
      "nvjpeg_decode_params_set_scale_factor",
      [](const NvjpegDecodeParams &p, nvjpegScaleFactor_t f) {
        return nvjpegDecodeParamsSetScaleFactor(p.params, f);
      },
      "decode_params"_a, "scale_factor"_a, Release());
  m.def(
      "nvjpeg_decode_params_set_exif_orientation",
      [](const NvjpegDecodeParams &p, nvjpegExifOrientation_t o) {
        return nvjpegDecodeParamsSetExifOrientation(p.params, o);
      },
      "decode_params"_a, "orientation"_a, Release());

  // -------------------------------------------------------------------------
  // Decoder (decoupled decode API)
  // -------------------------------------------------------------------------
  m.def(
      "nvjpeg_decoder_create",
      [](const NvjpegHandle &handle, nvjpegBackend_t implementation) {
        NvjpegJpegDecoder d;
        nvjpegStatus_t s = nvjpegDecoderCreate(handle.handle, implementation, &d.decoder);
        return std::make_tuple(s, d);
      },
      "nvjpeg_handle"_a, "implementation"_a, Release());
  m.def(
      "nvjpeg_decoder_destroy",
      [](NvjpegJpegDecoder &d) {
        nvjpegStatus_t s = nvjpegDecoderDestroy(d.decoder);
        if (s == NVJPEG_STATUS_SUCCESS) d.decoder = nullptr;
        return s;
      },
      "decoder_handle"_a, Release());
  m.def(
      "nvjpeg_decoder_jpeg_supported",
      [](const NvjpegJpegDecoder &d, const NvjpegJpegStream &js,
         const NvjpegDecodeParams &p) {
        int is_supported = -1;
        nvjpegStatus_t s =
            nvjpegDecoderJpegSupported(d.decoder, js.stream, p.params, &is_supported);
        return std::make_tuple(s, is_supported);
      },
      "decoder_handle"_a, "jpeg_stream"_a, "decode_params"_a, Release(),
      "Returns (status, is_supported); is_supported == 0 means supported.");
  m.def(
      "nvjpeg_decode_batched_supported",
      [](const NvjpegHandle &handle, const NvjpegJpegStream &js) {
        int is_supported = -1;
        nvjpegStatus_t s =
            nvjpegDecodeBatchedSupported(handle.handle, js.stream, &is_supported);
        return std::make_tuple(s, is_supported);
      },
      "handle"_a, "jpeg_stream"_a, Release(),
      "Returns (status, is_supported); is_supported == 0 means supported.");
  m.def(
      "nvjpeg_decode_batched_supported_ex",
      [](const NvjpegHandle &handle, const NvjpegJpegStream &js,
         const NvjpegDecodeParams &p) {
        int is_supported = -1;
        nvjpegStatus_t s = nvjpegDecodeBatchedSupportedEx(handle.handle, js.stream,
                                                          p.params, &is_supported);
        return std::make_tuple(s, is_supported);
      },
      "handle"_a, "jpeg_stream"_a, "decode_params"_a, Release(),
      "Returns (status, is_supported); is_supported == 0 means supported.");
  m.def(
      "nvjpeg_decoder_state_create",
      [](const NvjpegHandle &handle, const NvjpegJpegDecoder &d) {
        NvjpegJpegState st;
        nvjpegStatus_t s = nvjpegDecoderStateCreate(handle.handle, d.decoder, &st.state);
        return std::make_tuple(s, st);
      },
      "nvjpeg_handle"_a, "decoder_handle"_a, Release(),
      "Destroy the returned state with nvjpeg_jpeg_state_destroy.");

  m.def(
      "nvjpeg_decode_jpeg",
      [](const NvjpegHandle &handle, const NvjpegJpegDecoder &decoder,
         const NvjpegJpegState &decoder_state, const NvjpegJpegStream &jpeg_bitstream,
         NvjpegImage &destination, const NvjpegDecodeParams &decode_params,
         uintptr_t stream) {
        return nvjpegDecodeJpeg(handle.handle, decoder.decoder, decoder_state.state,
                                jpeg_bitstream.stream, &destination.image,
                                decode_params.params, as_stream(stream));
      },
      "handle"_a, "decoder"_a, "decoder_state"_a, "jpeg_bitstream"_a, "destination"_a,
      "decode_params"_a, "stream"_a = 0, Release());
  m.def(
      "nvjpeg_decode_jpeg_host",
      [](const NvjpegHandle &handle, const NvjpegJpegDecoder &decoder,
         const NvjpegJpegState &decoder_state, const NvjpegDecodeParams &decode_params,
         const NvjpegJpegStream &jpeg_stream) {
        return nvjpegDecodeJpegHost(handle.handle, decoder.decoder, decoder_state.state,
                                    decode_params.params, jpeg_stream.stream);
      },
      "handle"_a, "decoder"_a, "decoder_state"_a, "decode_params"_a, "jpeg_stream"_a,
      Release());
  m.def(
      "nvjpeg_decode_jpeg_transfer_to_device",
      [](const NvjpegHandle &handle, const NvjpegJpegDecoder &decoder,
         const NvjpegJpegState &decoder_state, const NvjpegJpegStream &jpeg_stream,
         uintptr_t stream) {
        return nvjpegDecodeJpegTransferToDevice(handle.handle, decoder.decoder,
                                                decoder_state.state, jpeg_stream.stream,
                                                as_stream(stream));
      },
      "handle"_a, "decoder"_a, "decoder_state"_a, "jpeg_stream"_a, "stream"_a = 0,
      Release());
  m.def(
      "nvjpeg_decode_jpeg_device",
      [](const NvjpegHandle &handle, const NvjpegJpegDecoder &decoder,
         const NvjpegJpegState &decoder_state, NvjpegImage &destination,
         uintptr_t stream) {
        return nvjpegDecodeJpegDevice(handle.handle, decoder.decoder, decoder_state.state,
                                      &destination.image, as_stream(stream));
      },
      "handle"_a, "decoder"_a, "decoder_state"_a, "destination"_a, "stream"_a = 0,
      Release());

  m.def(
      "nvjpeg_decode_batched_ex",
      [](const NvjpegHandle &handle, const NvjpegJpegState &jpeg_handle,
         nb::sequence data, std::optional<std::vector<size_t>> lengths,
         const std::vector<NvjpegImage *> &destinations,
         const std::vector<NvjpegDecodeParams *> &decode_params, uintptr_t stream) {
        auto items = host_data_list(data, lengths);
        if (destinations.size() != items.size() || decode_params.size() != items.size())
          throw nb::value_error(
              "`destinations` and `decode_params` must have the same length as `data`");
        std::vector<const unsigned char *> ptrs;
        std::vector<size_t> lens;
        std::vector<nvjpegImage_t> dst;
        std::vector<nvjpegDecodeParams_t> prm;
        for (size_t i = 0; i < items.size(); ++i) {
          ptrs.push_back(items[i].ptr());
          lens.push_back(items[i].len());
          dst.push_back(destinations[i]->image);
          prm.push_back(decode_params[i]->params);
        }
        nb::gil_scoped_release r;
        return nvjpegDecodeBatchedEx(handle.handle, jpeg_handle.state, ptrs.data(),
                                     lens.data(), dst.data(), prm.data(),
                                     as_stream(stream));
      },
      "handle"_a, "jpeg_handle"_a, "data"_a, "lengths"_a.none(), "destinations"_a,
      "decode_params"_a, "stream"_a = 0);
}
