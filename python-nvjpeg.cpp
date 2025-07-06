#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nvjpeg.h>

namespace nb = nanobind;

nvjpegStatus_t nvjpeg_create_simple(nvjpegHandle_t &handle) {
  return nvjpegCreateSimple(&handle);
}

nvjpegStatus_t nvjpeg_destroy(nvjpegHandle_t handle) {
  return nvjpegDestroy(handle);
}

nvjpegStatus_t nvjpeg_encoder_state_create(nvjpegHandle_t handle,
                                           nvjpegEncoderState_t &state,
                                           cudaStream_t stream) {
  return nvjpegEncoderStateCreate(handle, &state, stream);
}

nvjpegStatus_t nvjpeg_encoder_state_destroy(nvjpegEncoderState_t state) {
  return nvjpegEncoderStateDestroy(state);
}

nvjpegStatus_t nvjpeg_encoder_params_create(nvjpegHandle_t handle,
                                            nvjpegEncoderParams_t &params,
                                            cudaStream_t stream) {
  return nvjpegEncoderParamsCreate(handle, &params, stream);
}

nvjpegStatus_t nvjpeg_encoder_params_destroy(nvjpegEncoderParams_t params) {
  return nvjpegEncoderParamsDestroy(params);
}

nvjpegStatus_t
nvjpeg_encode_image(nvjpegHandle_t handle, nvjpegEncoderState_t state,
                    nvjpegEncoderParams_t params, const nvjpegImage_t &source,
                    nvjpegInputFormat_t input_format, int image_width,
                    int image_height, cudaStream_t stream) {
  return nvjpegEncodeImage(handle, state, params, &source, input_format,
                           image_width, image_height, stream);
}

void nvjpeg_image_set_channel(nvjpegImage_t &image, int channel_idx,
                              uintptr_t ptr, size_t pitch) {
  if (channel_idx >= 0 && channel_idx < NVJPEG_MAX_COMPONENT) {
    image.channel[channel_idx] = reinterpret_cast<unsigned char *>(ptr);
    image.pitch[channel_idx] = pitch;
  }
}

uintptr_t nvjpeg_image_get_channel_ptr(const nvjpegImage_t &image,
                                       int channel_idx) {
  if (channel_idx >= 0 && channel_idx < NVJPEG_MAX_COMPONENT) {
    return reinterpret_cast<uintptr_t>(image.channel[channel_idx]);
  }
  return 0;
}

size_t nvjpeg_image_get_pitch(const nvjpegImage_t &image, int channel_idx) {
  if (channel_idx >= 0 && channel_idx < NVJPEG_MAX_COMPONENT) {
    return image.pitch[channel_idx];
  }
  return 0;
}

nvjpegStatus_t nvjpeg_encode_get_buffer_size(nvjpegHandle_t handle,
                                             nvjpegEncoderParams_t params,
                                             int image_width, int image_height,
                                             size_t &max_stream_length) {
  return nvjpegEncodeGetBufferSize(handle, params, image_width, image_height,
                                   &max_stream_length);
}

nvjpegStatus_t nvjpeg_encode_retrieve_bitstream_size(nvjpegHandle_t handle,
                                                     nvjpegEncoderState_t state,
                                                     size_t &length,
                                                     cudaStream_t stream) {
  return nvjpegEncodeRetrieveBitstream(handle, state, nullptr, &length, stream);
}

nvjpegStatus_t nvjpeg_encode_retrieve_bitstream(nvjpegHandle_t handle,
                                                nvjpegEncoderState_t state,
                                                unsigned char *data,
                                                size_t &length,
                                                cudaStream_t stream) {
  return nvjpegEncodeRetrieveBitstream(handle, state, data, &length, stream);
}

nvjpegStatus_t nvjpeg_encoder_params_set_quality(nvjpegEncoderParams_t params,
                                                 int quality,
                                                 cudaStream_t stream) {
  return nvjpegEncoderParamsSetQuality(params, quality, stream);
}

nvjpegStatus_t nvjpeg_encoder_params_set_sampling_factors(
    nvjpegEncoderParams_t params, nvjpegChromaSubsampling_t chroma_subsampling,
    cudaStream_t stream) {
  return nvjpegEncoderParamsSetSamplingFactors(params, chroma_subsampling,
                                               stream);
}

NB_MODULE(pynvjpeg, m) {
  m.doc() = "a very thin wrapper for nvjpeg library";

  nb::class_<nvjpegHandle_t>(m, "nvjpegHandle_t").def(nb::init<>());

  nb::class_<nvjpegEncoderState_t>(m, "nvjpegEncoderState_t").def(nb::init<>());

  nb::class_<nvjpegEncoderParams_t>(m, "nvjpegEncoderParams_t")
      .def(nb::init<>());

  nb::class_<nvjpegImage_t>(m, "nvjpegImage_t").def(nb::init<>());

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
      .value("YUV", NVJPEG_INPUT_YUV)
      .value("RGB", NVJPEG_INPUT_RGB)
      .value("BGR", NVJPEG_INPUT_BGR)
      .value("RGBI", NVJPEG_INPUT_RGBI)
      .value("BGRI", NVJPEG_INPUT_BGRI)
      .value("NV12", NVJPEG_INPUT_NV12);

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
