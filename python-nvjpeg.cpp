#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nvjpeg.h>

namespace nb = nanobind;

struct NvjpegHandle {
  nvjpegHandle_t handle;
};

struct NvjpegEncoderState {
  nvjpegEncoderState_t state;
};

struct NvjpegEncoderParams {
  nvjpegEncoderParams_t params;
};

struct NvjpegImage {
  nvjpegImage_t image;
};

nvjpegStatus_t nvjpeg_create_simple(NvjpegHandle &handle) {
  return nvjpegCreateSimple(&handle.handle);
}

nvjpegStatus_t nvjpeg_destroy(NvjpegHandle &handle) {
  return nvjpegDestroy(handle.handle);
}

nvjpegStatus_t nvjpeg_encoder_state_create(const NvjpegHandle &handle,
                                           NvjpegEncoderState &state,
                                           uintptr_t stream) {
  return nvjpegEncoderStateCreate(handle.handle, &state.state,
                                  reinterpret_cast<cudaStream_t>(stream));
}

nvjpegStatus_t nvjpeg_encoder_state_destroy(NvjpegEncoderState &state) {
  return nvjpegEncoderStateDestroy(state.state);
}

nvjpegStatus_t nvjpeg_encoder_params_create(const NvjpegHandle &handle,
                                            NvjpegEncoderParams &params,
                                            uintptr_t stream) {
  return nvjpegEncoderParamsCreate(handle.handle, &params.params,
                                   reinterpret_cast<cudaStream_t>(stream));
}

nvjpegStatus_t nvjpeg_encoder_params_destroy(NvjpegEncoderParams &params) {
  return nvjpegEncoderParamsDestroy(params.params);
}

nvjpegStatus_t
nvjpeg_encode_image(const NvjpegHandle &handle, const NvjpegEncoderState &state,
                    const NvjpegEncoderParams &params,
                    const NvjpegImage &source, nvjpegInputFormat_t input_format,
                    int image_width, int image_height, uintptr_t stream) {
  return nvjpegEncodeImage(
      handle.handle, state.state, params.params, &source.image, input_format,
      image_width, image_height, reinterpret_cast<cudaStream_t>(stream));
}

void nvjpeg_image_set_channel(NvjpegImage &image, int channel_idx,
                              uintptr_t ptr, size_t pitch) {
  if (channel_idx >= 0 && channel_idx < NVJPEG_MAX_COMPONENT) {
    image.image.channel[channel_idx] = reinterpret_cast<unsigned char *>(ptr);
    image.image.pitch[channel_idx] = pitch;
  }
}

uintptr_t nvjpeg_image_get_channel_ptr(const NvjpegImage &image,
                                       int channel_idx) {
  if (channel_idx >= 0 && channel_idx < NVJPEG_MAX_COMPONENT) {
    return reinterpret_cast<uintptr_t>(image.image.channel[channel_idx]);
  }
  return 0;
}

size_t nvjpeg_image_get_pitch(const NvjpegImage &image, int channel_idx) {
  if (channel_idx >= 0 && channel_idx < NVJPEG_MAX_COMPONENT) {
    return image.image.pitch[channel_idx];
  }
  return 0;
}

nvjpegStatus_t nvjpeg_encode_get_buffer_size(const NvjpegHandle &handle,
                                             const NvjpegEncoderParams &params,
                                             int image_width, int image_height,
                                             size_t &max_stream_length) {
  return nvjpegEncodeGetBufferSize(handle.handle, params.params, image_width,
                                   image_height, &max_stream_length);
}

nvjpegStatus_t
nvjpeg_encode_retrieve_bitstream_size(const NvjpegHandle &handle,
                                      const NvjpegEncoderState &state,
                                      size_t &length, uintptr_t stream) {
  return nvjpegEncodeRetrieveBitstream(handle.handle, state.state, nullptr,
                                       &length,
                                       reinterpret_cast<cudaStream_t>(stream));
}

nvjpegStatus_t nvjpeg_encode_retrieve_bitstream(const NvjpegHandle &handle,
                                                const NvjpegEncoderState &state,
                                                unsigned char *data,
                                                size_t &length,
                                                uintptr_t stream) {
  return nvjpegEncodeRetrieveBitstream(handle.handle, state.state, data,
                                       &length,
                                       reinterpret_cast<cudaStream_t>(stream));
}

nvjpegStatus_t
nvjpeg_encoder_params_set_quality(const NvjpegEncoderParams &params,
                                  int quality, uintptr_t stream) {
  return nvjpegEncoderParamsSetQuality(params.params, quality,
                                       reinterpret_cast<cudaStream_t>(stream));
}

nvjpegStatus_t nvjpeg_encoder_params_set_sampling_factors(
    const NvjpegEncoderParams &params,
    nvjpegChromaSubsampling_t chroma_subsampling, uintptr_t stream) {
  return nvjpegEncoderParamsSetSamplingFactors(
      params.params, chroma_subsampling,
      reinterpret_cast<cudaStream_t>(stream));
}

NB_MODULE(nvjpeg_wrapper, m) {
  m.doc() = "Thin wrapper for nvjpeg library";

  nb::class_<NvjpegHandle>(m, "NvjpegHandle").def(nb::init<>());

  nb::class_<NvjpegEncoderState>(m, "NvjpegEncoderState").def(nb::init<>());

  nb::class_<NvjpegEncoderParams>(m, "NvjpegEncoderParams").def(nb::init<>());

  nb::class_<NvjpegImage>(m, "NvjpegImage").def(nb::init<>());

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
      .value("IMPLEMENTATION_NOT_SUPPORTED",
             NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED);

  nb::enum_<nvjpegInputFormat_t>(m, "nvjpegInputFormat_t")
      .value("RGB", NVJPEG_INPUT_RGB)
      .value("BGR", NVJPEG_INPUT_BGR)
      .value("RGBI", NVJPEG_INPUT_RGBI)
      .value("BGRI", NVJPEG_INPUT_BGRI);

  nb::enum_<nvjpegChromaSubsampling_t>(m, "nvjpegChromaSubsampling_t")
      .value("CSS_444", NVJPEG_CSS_444)
      .value("CSS_422", NVJPEG_CSS_422)
      .value("CSS_420", NVJPEG_CSS_420)
      .value("CSS_440", NVJPEG_CSS_440)
      .value("CSS_411", NVJPEG_CSS_411)
      .value("CSS_410", NVJPEG_CSS_410)
      .value("CSS_GRAY", NVJPEG_CSS_GRAY)
      .value("CSS_UNKNOWN", NVJPEG_CSS_UNKNOWN);

  m.def("nvjpeg_create_simple", &nvjpeg_create_simple);
  m.def("nvjpeg_destroy", &nvjpeg_destroy);
  m.def("nvjpeg_encoder_state_create", &nvjpeg_encoder_state_create);
  m.def("nvjpeg_encoder_state_destroy", &nvjpeg_encoder_state_destroy);
  m.def("nvjpeg_encoder_params_create", &nvjpeg_encoder_params_create);
  m.def("nvjpeg_encoder_params_destroy", &nvjpeg_encoder_params_destroy);
  m.def("nvjpeg_encode_image", &nvjpeg_encode_image);
  m.def("nvjpeg_image_set_channel", &nvjpeg_image_set_channel);
  m.def("nvjpeg_image_get_channel_ptr", &nvjpeg_image_get_channel_ptr);
  m.def("nvjpeg_image_get_pitch", &nvjpeg_image_get_pitch);
  m.def("nvjpeg_encode_get_buffer_size", &nvjpeg_encode_get_buffer_size);
  m.def("nvjpeg_encode_retrieve_bitstream_size",
        &nvjpeg_encode_retrieve_bitstream_size);
  m.def("nvjpeg_encode_retrieve_bitstream", &nvjpeg_encode_retrieve_bitstream);
  m.def("nvjpeg_encoder_params_set_quality",
        &nvjpeg_encoder_params_set_quality);
  m.def("nvjpeg_encoder_params_set_sampling_factors",
        &nvjpeg_encoder_params_set_sampling_factors);

  m.attr("MAX_COMPONENT") = NVJPEG_MAX_COMPONENT;
}
