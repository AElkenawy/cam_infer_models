
#include <unistd.h>
#include <atomic>
#include <csignal>
#include <string>
#include <cstdlib>

extern "C" {
#include <pipewire/pipewire.h>
}

#include <spa/param/video/format-utils.h>
#include <spa/param/video/raw.h>
#include <spa/utils/result.h>
#include <spdlog/spdlog.h>

#include "interface.h"

#define YUY2_BYTES_PER_PIXEL 2
#define DEFAULT_BUFFERS 4
#define MIN_BUFFERS 2
#define MAX_BUFFERS 16

constexpr bool kVisible = true;
constexpr float kNmsThreshold = 0.4f;
constexpr float kConfThreshold = 0.7f;

static std::atomic<int> active_detections = 0;
static constexpr int MAX_CONCURRENT_DETECTIONS = 2;
auto dataPath = "/path/to/cam_infer_models/";
std::string yoloBasePath;
std::string yoloClassesFile;

uint32_t frame_width;
uint32_t frame_height;
bool detect_done = false;

static struct impl* g_impl = nullptr;

struct impl {
  pw_main_loop* loop;
  pw_context* context;
  pw_core* core;

  pw_stream* capture;
  pw_stream* raw_playback;
  pw_stream* detection_playback;

  struct spa_hook capture_listener;
  struct spa_hook raw_playback_listener;
  struct spa_hook detection_playback_listener;

  struct spa_video_info_raw capture_info;
  struct spa_video_info_raw raw_playback_info;
  struct spa_video_info_raw detection_playback_info;
};

static void copy_buffer(const pw_buffer* in, const pw_buffer* out) {
  assert(in->buffer->n_datas == out->buffer->n_datas);

  for (uint32_t i = 0; i < out->buffer->n_datas; i++) {
    const spa_data* src = &in->buffer->datas[i];
    const spa_data* dst = &out->buffer->datas[i];
    memcpy(dst->data, src->data, src->chunk->size);
    dst->chunk->offset = 0;
    dst->chunk->size = src->chunk->size;
    dst->chunk->stride = src->chunk->stride;
  }
}

/* Fixed signature: accept non-const `impl*` to match detection_callback_t */
static void detect_completed_callback(struct pw_buffer* buffer,
                                      struct impl* impl,
                                      bool success) {
  --active_detections;
  pw_stream_queue_buffer(impl->detection_playback, buffer);
  pw_stream_trigger_process(impl->detection_playback);
}

static void on_process(void* data) {
  auto impl = static_cast<struct impl*>(data);
  struct pw_buffer *in, *out;

  yoloBasePath = dataPath + std::string("yolo/");
  yoloClassesFile = yoloBasePath + std::string("coco.names");

  pw_log_debug("on_process called");

  if ((in = pw_stream_dequeue_buffer(impl->capture)) == nullptr) {
    pw_log_warn("no input buffer");
    return;
  }

  if ((out = pw_stream_dequeue_buffer(impl->raw_playback)) != nullptr) {
    pw_log_debug("copying to raw_playback stream");
    copy_buffer(in, out);
    pw_stream_queue_buffer(impl->raw_playback, out);
    pw_stream_trigger_process(impl->raw_playback);
  }

  if (active_detections < MAX_CONCURRENT_DETECTIONS) {
    if ((out = pw_stream_dequeue_buffer(impl->detection_playback)) != nullptr) {
      copy_buffer(in, out);
      ++active_detections;
      detectObjects_async(out, kConfThreshold, kNmsThreshold,
                          yoloBasePath.c_str(), yoloClassesFile.c_str(),
                          frame_width, frame_height, kVisible,
                          detect_completed_callback, impl);
    }
  }

  pw_stream_queue_buffer(impl->capture, in);
}

static void on_stream_state_changed(void* data,
                                    enum pw_stream_state old,
                                    enum pw_stream_state state,
                                    const char* error) {
  auto impl = static_cast<struct impl*>(data);

  printf("stream state changed: %s -> %s\n", pw_stream_state_as_string(old),
         pw_stream_state_as_string(state));

  if (error)
    printf(" (error: %s)", error);
  printf("\n");

  if (state == PW_STREAM_STATE_ERROR) {
    pw_main_loop_quit(impl->loop);
  }
}

static void on_capture_param_changed(void* data, uint32_t id,
                                     const struct spa_pod* param) {
  auto impl = static_cast<struct impl*>(data);
  struct spa_video_info_raw format = SPA_VIDEO_INFO_RAW_INIT(SPA_VIDEO_FORMAT_UNKNOWN);
  uint32_t size;
  const struct spa_pod* params[2];
  uint8_t buffer[1024];
  struct spa_pod_builder b{};

  if (param == nullptr || id != SPA_PARAM_Format)
    return;

  pw_log_info("format changed");
  spa_format_video_raw_parse(param, &format);
  frame_width = format.size.width;
  frame_height = format.size.height;

  size = SPA_ROUND_UP_N(format.size.width * format.size.height * YUY2_BYTES_PER_PIXEL, 4);

  // Configure buffers for the capture stream
  spa_pod_builder_init(&b, buffer, sizeof(buffer));

  params[0] = static_cast<const spa_pod*>(spa_pod_builder_add_object(&b,
      SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
      SPA_PARAM_BUFFERS_buffers,  SPA_POD_CHOICE_RANGE_Int(DEFAULT_BUFFERS, MIN_BUFFERS, MAX_BUFFERS),
      SPA_PARAM_BUFFERS_blocks,   SPA_POD_Int(1),
      SPA_PARAM_BUFFERS_size,     SPA_POD_Int(size),
      SPA_PARAM_BUFFERS_dataType, SPA_POD_CHOICE_FLAGS_Int(1<<SPA_DATA_MemFd)));

  pw_stream_update_params(impl->capture, params, 1);

  // Forward exact format and buffers to the output streams
  spa_pod_builder_init(&b, buffer, sizeof(buffer));

  params[0] = spa_format_video_raw_build(&b, SPA_PARAM_EnumFormat, &format);
  params[1] = static_cast<const spa_pod*>(spa_pod_builder_add_object(&b,
      SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
      SPA_PARAM_BUFFERS_buffers,  SPA_POD_CHOICE_RANGE_Int(DEFAULT_BUFFERS, MIN_BUFFERS, MAX_BUFFERS),
      SPA_PARAM_BUFFERS_blocks,   SPA_POD_Int(1),
      SPA_PARAM_BUFFERS_size,     SPA_POD_Int(size),
      SPA_PARAM_BUFFERS_dataType, SPA_POD_CHOICE_FLAGS_Int(1<<SPA_DATA_MemFd)));

  pw_stream_update_params(impl->raw_playback, params, 2);
  pw_stream_update_params(impl->detection_playback, params, 2);
}

static const struct pw_stream_events capture_stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
    .param_changed = on_capture_param_changed,
    .process = on_process,
};

static const struct pw_stream_events playback_stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
};

static void signal_handler(int signo) {
  if (signo == SIGINT) {
    fprintf(stderr, "\nCaught SIGINT, exiting...\n");
    if (g_impl != nullptr) {
      pw_main_loop_quit(g_impl->loop);
    }
  }
}

int main(int argc, char* argv[]) {
  const struct spa_pod* params[1];
  uint8_t buffer[1024];
  struct spa_pod_builder b{};
  int res;

  auto impl = std::make_unique<struct impl>();
  if (impl == nullptr)
    return -1;

  g_impl = impl.get();

  pw_init(&argc, &argv);

  impl->loop = pw_main_loop_new(nullptr);
  if (impl->loop == nullptr) {
    spdlog::error("Failed to create main loop");
    return -1;
  }

  impl->context = pw_context_new(pw_main_loop_get_loop(impl->loop), nullptr, 0);
  if (impl->context == nullptr) {
    spdlog::error("Failed to create context");
    return -1;
  }

  impl->core = pw_context_connect(impl->context, nullptr, 0);
  if (impl->core == nullptr) {
    spdlog::error("Failed to connect to PipeWire");
    return -1;
  }

  impl->capture_info = {.format = SPA_VIDEO_FORMAT_YUY2,
                        .size = {.width = 640, .height = 480}};
  impl->raw_playback_info = impl->capture_info;
  impl->detection_playback_info = impl->capture_info;

  spa_autofree char* link_group = spa_aprintf("camera-filter-%d", getpid());
  impl->capture = pw_stream_new(
      impl->core, "filter-capture",
      pw_properties_new(
          PW_KEY_MEDIA_TYPE, "Video",
          PW_KEY_MEDIA_CATEGORY, "Capture",
          PW_KEY_MEDIA_ROLE, "Camera",
          PW_KEY_NODE_DESCRIPTION, "camera sink",
          PW_KEY_MEDIA_CLASS, "Stream/Input/Video",
          PW_KEY_NODE_LINK_GROUP, link_group,
          NULL));

  impl->raw_playback = pw_stream_new(
      impl->core, "raw-playback",
      pw_properties_new(
          PW_KEY_MEDIA_TYPE, "Video",
          PW_KEY_MEDIA_CATEGORY, "Playback",
          PW_KEY_MEDIA_ROLE, "Camera",
          PW_KEY_NODE_DESCRIPTION, "raw playback",
          PW_KEY_NODE_NAME, "camera-raw-output",
          PW_KEY_MEDIA_CLASS, "Stream/Output/Video",
          PW_KEY_NODE_LINK_GROUP, link_group,
          NULL));

  impl->detection_playback = pw_stream_new(
      impl->core, "detection-playback",
      pw_properties_new(
          PW_KEY_MEDIA_TYPE, "Video",
          PW_KEY_MEDIA_CATEGORY, "Playback",
          PW_KEY_MEDIA_ROLE, "Camera",
          PW_KEY_NODE_DESCRIPTION, "detection playback",
          PW_KEY_NODE_NAME, "camera-detection-output",
          PW_KEY_MEDIA_CLASS, "Stream/Output/Video",
          PW_KEY_NODE_LINK_GROUP, link_group,
          NULL));

  pw_stream_add_listener(impl->capture,
                        &impl->capture_listener,
                        &capture_stream_events, impl.get());
  pw_stream_add_listener(impl->raw_playback,
                        &impl->raw_playback_listener,
                        &playback_stream_events, impl.get());
  pw_stream_add_listener(impl->detection_playback,
                        &impl->detection_playback_listener,
                        &playback_stream_events, impl.get());

  spa_pod_builder_init(&b, buffer, sizeof(buffer));
  params[0] =
      spa_format_video_raw_build(&b, SPA_PARAM_EnumFormat, &impl->capture_info);

  if ((res = pw_stream_connect(
           impl->capture, PW_DIRECTION_INPUT, PW_ID_ANY,
           static_cast<pw_stream_flags>(PW_STREAM_FLAG_AUTOCONNECT |
                                        PW_STREAM_FLAG_MAP_BUFFERS |
                                        PW_STREAM_FLAG_ASYNC),
           params, 1)) < 0) {
    spdlog::error("Failed to connect capture stream: {}", spa_strerror(res));
    return -1;
  }

  spa_pod_builder_init(&b, buffer, sizeof(buffer));
  params[0] = spa_format_video_raw_build(&b, SPA_PARAM_EnumFormat,
                                         &impl->raw_playback_info);

  if ((res = pw_stream_connect(
           impl->raw_playback, PW_DIRECTION_OUTPUT, PW_ID_ANY,
           static_cast<pw_stream_flags>(PW_STREAM_FLAG_MAP_BUFFERS |
                                        PW_STREAM_FLAG_TRIGGER |
                                        PW_STREAM_FLAG_ASYNC),
           params, 1)) < 0) {
    spdlog::error("Failed to connect raw playback stream: {}",
                  spa_strerror(res));
    return -1;
  }

  spa_pod_builder_init(&b, buffer, sizeof(buffer));
  params[0] = spa_format_video_raw_build(&b, SPA_PARAM_EnumFormat,
                                         &impl->detection_playback_info);

  if ((res = pw_stream_connect(
           impl->detection_playback, PW_DIRECTION_OUTPUT, PW_ID_ANY,
           static_cast<pw_stream_flags>(PW_STREAM_FLAG_MAP_BUFFERS |
                                        PW_STREAM_FLAG_TRIGGER |
                                        PW_STREAM_FLAG_ASYNC),
           params, 1)) < 0) {
    spdlog::error("Failed to connect detection playback stream: {}",
                  spa_strerror(res));
    return -1;
  }

  signal(SIGINT, signal_handler);

  spdlog::info("Running... Press Ctrl+C to exit");
  pw_main_loop_run(impl->loop);

  pw_stream_destroy(impl->capture);
  pw_stream_destroy(impl->raw_playback);
  pw_stream_destroy(impl->detection_playback);
  pw_context_destroy(impl->context);
  pw_main_loop_destroy(impl->loop);
  pw_deinit();

  g_impl = nullptr;

  return EXIT_SUCCESS;
}
