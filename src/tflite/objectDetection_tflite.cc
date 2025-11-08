
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/string_util.h>

#include "interface.h"

bool tensor_loaded = false;
std::unique_ptr<tflite::Interpreter> interpreter;
std::unique_ptr<tflite::FlatBufferModel> model;
std::mutex tensor_mutex;

bool enable_profiling = true;
auto async_file =
    spdlog::basic_logger_mt<spdlog::async_factory>("async_file_logger",
                                                   "logs/async_log.txt");

// Structure to pass data to async thread
struct AsyncDetectionData {
  struct pw_buffer* buffer;
  float confThreshold;
  float nmsThreshold;
  std::string basePath;
  std::string classesFile;
  uint32_t frame_width;
  uint32_t frame_height;
  bool bVis;
  detection_callback_t callback;
  struct impl* user_data;
};

// Object Detection Results Structure
struct Detection {
  float confidence;
  int class_id;
  cv::Rect bbox;
};

bool initialize_tf(const std::string& model_path) {
  spdlog::set_default_logger(async_file);
  std::lock_guard lock(tensor_mutex);
  if (tensor_loaded) {
    return true;
  }
  try {
    const auto t1_start = std::chrono::system_clock::now();

    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
      std::cerr << "Error: failed to load model: " << model_path << std::endl;
      return false;
    }

    // Build interpreter from the global model
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interp;
    tflite::InterpreterBuilder(*model, resolver)(&interp);

    if (!interp) {
      std::cerr << "Failed to create interpreter" << std::endl;
      return false;
    }

    if (interp->AllocateTensors() != kTfLiteOk) {
      std::cerr << "Failed to allocate tensors" << std::endl;
      return false;
    }

    // Configure interpreter
    interp->SetAllowFp16PrecisionForFp32(true);
    interp->SetNumThreads(1);

    // Move the local interpreter to the global variable
    interpreter = std::move(interp);
    tensor_loaded = true;

    const auto t1_end = std::chrono::system_clock::now();
    const auto t1_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t1_end - t1_start)
            .count();
    if (enable_profiling)
      spdlog::info("Interpreter initialized in {} us", t1_duration);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Exception initializing interpreter: " << e.what()
              << std::endl;
    return false;
  }
}

std::unique_ptr<tflite::Interpreter>& get_interpreter() {
  return interpreter;
}

cv::Mat pwbuffer_to_cvmat(struct pw_buffer* buf,
                          uint32_t frame_width,
                          uint32_t frame_height) {
  static int t2_count = 0;
  static long long t2_total_duration = 0;
  const auto t2_start = std::chrono::steady_clock::now();

  if (!buf || buf->buffer->n_datas == 0)
    return {};
  struct spa_data* spa_data = &buf->buffer->datas[0];
  auto* data = static_cast<uint8_t*>(spa_data->data);

  cv::Mat yuy2_frame(frame_height, frame_width, CV_8UC2, data);

  cv::Mat rgb_frame;
  cv::cvtColor(yuy2_frame, rgb_frame, cv::COLOR_YUV2BGR_YUY2);

  const auto t2_end = std::chrono::steady_clock::now();
  const auto t2_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2_end - t2_start)
          .count();
  t2_total_duration += t2_duration;
  t2_count++;
  if (t2_count == 10) {
    double t2_average = static_cast<double>(t2_total_duration) / t2_count;
    if (enable_profiling)
      spdlog::info("pwbuffer_to_cvmat {} cycles average duration: {} us",
                   t2_count, t2_average);
    t2_count = 0;
    t2_total_duration = 0;
  }
  return rgb_frame;
}

void cvmat_to_pwbuffer(cv::Mat visImg,
                       struct pw_buffer* buf,
                       uint32_t frame_width,
                       uint32_t frame_height) {
  if (!buf || visImg.empty())
    return;

  static int t3_count = 0;
  static long long t3_total_duration = 0;
  const auto t3_start = std::chrono::steady_clock::now();

  cv::Mat yuy2_frame;
  cv::resize(visImg, visImg, cv::Size(frame_width, frame_height));
  cv::cvtColor(visImg, yuy2_frame, cv::COLOR_BGR2YUV_YUY2);

  struct spa_data* spa_data = &buf->buffer->datas[0];
  auto* data = static_cast<uint8_t*>(spa_data->data);

  const size_t bytes_to_copy = frame_width * frame_height * 2;

  if (yuy2_frame.total() * yuy2_frame.elemSize() >= bytes_to_copy &&
      spa_data->maxsize >= bytes_to_copy) {
    memcpy(data, yuy2_frame.data, bytes_to_copy);
  }
  const auto t3_end = std::chrono::steady_clock::now();
  const auto t3_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t3_end - t3_start)
          .count();
  t3_total_duration += t3_duration;
  t3_count++;
  if (t3_count == 10) {
    double t3_average = static_cast<double>(t3_total_duration) / t3_count;
    if (enable_profiling)
      spdlog::info("cvmat_to_pwbuffer {} cycles average duration: {} us",
                   t3_count, t3_average);
    t3_count = 0;
    t3_total_duration = 0;
  }
}

bool initialize_tf_once(const std::string& model_path) {
  static std::once_flag init_flag;
  static bool init_success = false;

  std::call_once(init_flag,
                 [model_path]() { init_success = initialize_tf(model_path); });

  return init_success;
}

// Non-Maximum Suppression function
std::vector<Detection> applyNMS(const std::vector<Detection>& detections,
                                float nmsThreshold) {
  std::vector<Detection> filtered_detections;
  std::vector<bool> suppressed(detections.size(), false);

  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i])
      continue;

    filtered_detections.push_back(detections[i]);

    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (suppressed[j])
        continue;

      // Calculate IoU
      const cv::Rect& box1 = detections[i].bbox;
      const cv::Rect& box2 = detections[j].bbox;

      int x1 = std::max(box1.x, box2.x);
      int y1 = std::max(box1.y, box2.y);
      int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
      int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

      if (x1 < x2 && y1 < y2) {
        int intersection_area = (x2 - x1) * (y2 - y1);
        int union_area = box1.width * box1.height + box2.width * box2.height -
                         intersection_area;

        if (float iou = static_cast<float>(intersection_area) / union_area;
            iou > nmsThreshold) {
          suppressed[j] = true;
        }
      }
    }
  }

  return filtered_detections;
}

void detectobjects_worker_thread(AsyncDetectionData* data) {
  bool success = false;

  try {
    cv::Mat img =
        pwbuffer_to_cvmat(data->buffer, data->frame_width, data->frame_height);

    std::string tflite_model = std::string(data->basePath) + "ssd_mobilenet.tflite";

    if (!initialize_tf_once(tflite_model)) {
      std::cerr << "Failed to initialize network" << std::endl;
      data->callback(data->buffer, data->user_data, false);
      delete data;
      return;
    }

    // Cache classes loading (load once per thread or globally)
    thread_local std::vector<std::string> cached_classes;
    thread_local std::string cached_classes_file;

    if (cached_classes_file != data->classesFile) {
      cached_classes.clear();
      std::ifstream ifs(data->classesFile);
      std::string line;
      while (getline(ifs, line)) {
        cached_classes.push_back(line);
      }
      cached_classes_file = data->classesFile;
    }

    cv::Size originalSize = img.size();

    std::lock_guard lock(tensor_mutex);

    auto& interp = get_interpreter();

    if (!interp) {
      std::cerr << "Interpreter is null! Model not initialized." << std::endl;
      delete data;
      return;
    }

    // Get Input Tensor Dimensions
    int input = interp->inputs()[0];
    TfLiteTensor* input_tensor = interp->tensor(input);
    if (!input_tensor) {
      std::cerr << "Input tensor is null!" << std::endl;
      delete data;
      return;
    }

    auto height = input_tensor->dims->data[1];
    auto width = input_tensor->dims->data[2];

    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(width, height), cv::INTER_LINEAR);

    // Ensure the Mat is contiguous and has the right type
    if (!resized_img.isContinuous() || resized_img.type() != CV_8UC3) {
      resized_img = resized_img.clone();
    }

    // Validate sizes
    size_t tensor_bytes = input_tensor->bytes;
    size_t img_bytes = resized_img.total() * resized_img.elemSize();
    if (img_bytes != tensor_bytes) {
      spdlog::error("Image size ({}) != tensor size ({}). Cannot proceed.",
                    img_bytes, tensor_bytes);
      delete data;
      return;
    }

    // Copy image data into the model input tensor
    memcpy(interp->typed_input_tensor<unsigned char>(0), resized_img.data,
           tensor_bytes);

    const auto tinv_start = std::chrono::steady_clock::now();
    if (interp->Invoke() != kTfLiteOk) {
      std::cerr << "Error: interpreter Invoke() failed" << std::endl;
      delete data;
      return;
    }
    const auto tinv_end = std::chrono::steady_clock::now();
    const auto tinv_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(tinv_end -
                                                              tinv_start)
            .count();
    if (enable_profiling)
      spdlog::info("invoke duration: {} us", tinv_duration);

    // Get Output - SSD MobileNet with 4 output tensors
    int num_outputs = interp->outputs().size();
    spdlog::info("Model has {} output tensors", num_outputs);

    std::vector<Detection> detections;

    if (num_outputs == 4) {
      // Standard SSD MobileNet format with 4 outputs:
      // [0] = detection_boxes [1, num_detections, 4] - normalized coordinates
      // [1] = detection_classes [1, num_detections] - class indices
      // [2] = detection_scores [1, num_detections] - confidence scores
      // [3] = num_detections [1] - number of valid detections

      TfLiteTensor* boxes_tensor = interp->tensor(interp->outputs()[0]);
      TfLiteTensor* classes_tensor = interp->tensor(interp->outputs()[1]);
      TfLiteTensor* scores_tensor = interp->tensor(interp->outputs()[2]);
      TfLiteTensor* num_tensor = interp->tensor(interp->outputs()[3]);

      // Log tensor shapes for debugging
      spdlog::info("Output 0 (boxes): dims={}", boxes_tensor->dims->size);
      for (int i = 0; i < boxes_tensor->dims->size; ++i) {
        spdlog::info("  boxes dim[{}] = {}", i, boxes_tensor->dims->data[i]);
      }
      spdlog::info("Output 1 (classes): type={}",
                   static_cast<int>(classes_tensor->type));
      spdlog::info("Output 2 (scores): type={}",
                   static_cast<int>(scores_tensor->type));
      spdlog::info("Output 3 (num_detections): type={}",
                   static_cast<int>(num_tensor->type));

      // All outputs should be float32
      const float* boxes = interp->typed_output_tensor<float>(0);
      const float* classes = interp->typed_output_tensor<float>(1);
      const float* scores = interp->typed_output_tensor<float>(2);
      const float* num_det = interp->typed_output_tensor<float>(3);

      int num_detections = static_cast<int>(num_det[0]);
      spdlog::info("Number of detections: {}", num_detections);

      for (int i = 0; i < num_detections; ++i) {
        if (float confidence = scores[i]; confidence > data->confThreshold) {
          int class_id = static_cast<int>(classes[i]);

          // Boxes are in format [ymin, xmin, ymax, xmax], normalized [0-1]
          float ymin = boxes[i * 4 + 0] * originalSize.height;
          float xmin = boxes[i * 4 + 1] * originalSize.width;
          float ymax = boxes[i * 4 + 2] * originalSize.height;
          float xmax = boxes[i * 4 + 3] * originalSize.width;

          // Clamp to image boundaries
          xmin = std::max(
              0.0f, std::min(xmin, static_cast<float>(originalSize.width)));
          ymin = std::max(
              0.0f, std::min(ymin, static_cast<float>(originalSize.height)));
          xmax = std::max(
              0.0f, std::min(xmax, static_cast<float>(originalSize.width)));
          ymax = std::max(
              0.0f, std::min(ymax, static_cast<float>(originalSize.height)));

          spdlog::info(
              "Detection {}: class={}, conf={:.2f}, "
              "box=[{:.0f},{:.0f},{:.0f},{:.0f}]",
              i, class_id, confidence, xmin, ymin, xmax, ymax);

          detections.push_back(
              {confidence, class_id,
               cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax))});
        }
      }
    } else {
      spdlog::error("Expected 4 output tensors, got {}", num_outputs);

      // Log all output shapes for debugging
      for (int idx = 0; idx < num_outputs; ++idx) {
        int output_idx = interp->outputs()[idx];
        TfLiteTensor* tensor = interp->tensor(output_idx);
        spdlog::info("Output [{}]: type={}, dims={}", idx,
                     static_cast<int>(tensor->type), tensor->dims->size);
        for (int d = 0; d < tensor->dims->size; ++d) {
          spdlog::info("    dim[{}] = {}", d, tensor->dims->data[d]);
        }
      }

      delete data;
      return;
    }

    spdlog::info("Total detections before NMS: {}", detections.size());

    // Apply Non-Maximum Suppression
    std::vector<Detection> filtered_detections =
        applyNMS(detections, data->nmsThreshold);

    spdlog::info("After NMS: {} detections", filtered_detections.size());

    // Create the visualization image (clone original for drawing)
    cv::Mat visImg = img.clone();

    // Draw bounding boxes and labels on the original size image
    for (const auto& [confidence, class_id, bbox] : filtered_detections) {
      if (class_id >= 0 &&
          class_id < static_cast<int>(cached_classes.size())) {
        // Draw bounding box
        cv::rectangle(visImg, bbox, cv::Scalar(0, 255, 0), 2);

        // Draw label background for better readability
        std::string label =
            cached_classes[class_id] + " " +
            std::to_string(static_cast<int>(confidence * 100)) + "%";

        int baseline = 0;
        cv::Size labelSize =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        int label_y = std::max(bbox.y, labelSize.height);
        cv::rectangle(
            visImg, cv::Point(bbox.x, label_y - labelSize.height),
            cv::Point(bbox.x + labelSize.width, label_y + baseline),
            cv::Scalar(0, 255, 0), cv::FILLED);

        cv::putText(visImg, label, cv::Point(bbox.x, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
      }
    }

    cvmat_to_pwbuffer(visImg, data->buffer, data->frame_width,
                      data->frame_height);
    success = true;
  } catch (const std::exception& e) {
    spdlog::error("Detection failed: {}", e.what());
    success = false;
  }

  data->callback(data->buffer, data->user_data, success);

  delete data;
}

// Async entry point
void detectObjects_async(struct pw_buffer* out_buffer,
                         float confThreshold,
                         float nmsThreshold,
                         const char* basePath,
                         const char* classesFile,
                         uint32_t frame_width,
                         uint32_t frame_height,
                         bool bVis,
                         detection_callback_t callback,
                         struct impl* user_data) {
  // thread data structure
  auto data =
      new AsyncDetectionData{.buffer = out_buffer,
                             .confThreshold = confThreshold,
                             .nmsThreshold = nmsThreshold,
                             .basePath = std::string(basePath),
                             .classesFile = std::string(classesFile),
                             .frame_width = frame_width,
                             .frame_height = frame_height,
                             .bVis = bVis,
                             .callback = callback,
                             .user_data = user_data};

  std::thread detection_thread(detectobjects_worker_thread, data);
  detection_thread.detach();
}
