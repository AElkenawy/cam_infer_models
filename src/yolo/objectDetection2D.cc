// Based on:
// https://github.com/Geekgineer/YOLOs-CPP/blob/main/include/det/YOLO11.hpp
// Adapted for use in cam_infer_models project

#include <algorithm>
#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include "dataStructures.h"

#include "interface.h"

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <chrono>
#include <functional>
#include <thread>
#include "spdlog/spdlog.h"

// gRPC includes for model switching
#include <grpcpp/grpcpp.h>
#include "coordinator.grpc.pb.h"

using namespace std;

// neural network instance
cv::dnn::Net net;
bool n_net_loaded = false;
std::mutex n_net_mutex;

bool enable_profiling = true;
auto async_file =
    spdlog::basic_logger_mt<spdlog::async_factory>("async_file_logger",
                                                   "logs/async_log.txt");

// Global state for model switching
static bool g_should_reinitialize = false;
static bool g_should_stop = false;
static bool g_reinitializing_in_progress =
    false;                         // Prevent multiple simultaneous switches
static std::mutex g_switch_mutex;  // Mutex for switch operations
static int g_detection_count = 0;
static constexpr int CHECK_RESPONSE_INTERVAL =
    10;  // Check response every 10 detections

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

std::string get_model_path_for_current_model(const std::string& basePath) {
  if (const std::string current_model = get_current_model_grpc();
      current_model == "traffic_signs") {
    return basePath + "traffic_signs_yolo11s.onnx";
  } else if (current_model == "security_mode") {
    return basePath + "security_mode_yolo11s.onnx";
  } else {
    // Default fallback
    return basePath + "yolo11s.onnx";
  }
}

std::string get_class_file_for_current_model(const std::string& basePath) {
  if (const std::string current_model = get_current_model_grpc();
      current_model == "traffic_signs") {
    return basePath + "coco_traffic.names";
  } else if (current_model == "security_mode") {
    return basePath + "coco_security.names";
  }
  // Default fallback
  return basePath + "coco.names";
}

bool initialize_nn(const std::string& model_path) {
  spdlog::set_default_logger(async_file);
  std::lock_guard lock(n_net_mutex);
  if (n_net_loaded) {
    return true;
  }
  try {
    const auto t1_start = std::chrono::system_clock::now();

    // gRPC client initialization
    if (const bool grpc_connected = init_grpc_client("localhost:50051");
        !grpc_connected) {
      std::cout << "Failed to connect to gRPC coordinator, using default model"
                << std::endl;
      // Fallback to default model when no gRPC server available
      net = cv::dnn::readNetFromONNX(model_path);
    } else {
      const std::string selected_model = get_current_model_grpc();
      std::cout << "gRPC client connected successfully with model: "
                << selected_model << std::endl;

      const std::string actual_model_path = get_model_path_for_current_model(
          model_path.substr(0, model_path.find_last_of('/') + 1));
      std::cout << "Loading model: " << actual_model_path << std::endl;

      net = cv::dnn::readNetFromONNX(actual_model_path);

      if (net.empty()) {
        // Fallback to default model if selected model fails
        std::cout << "Selected model failed, trying default model..."
                  << std::endl;
        net = cv::dnn::readNetFromONNX(model_path);
      }
    }
#ifdef USE_CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#else
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
#endif

    if (net.empty()) {
      cerr << "Error: Network not loaded properly" << endl;
      return false;
    }
    n_net_loaded = true;
    const auto t1_end = std::chrono::system_clock::now();
    const auto t1_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t1_end - t1_start)
            .count();
    if (enable_profiling)
      spdlog::info("initialize_nn duration: {} us", t1_duration);
    return true;
  } catch (const cv::Exception& e) {
    std::cerr << "Error loading neural network: " << e.what() << endl;
    return false;
  }
}

// Force reinitialization of neural network (for model switching)
bool reinitialize_nn_force(const std::string& model_path) {
  spdlog::set_default_logger(async_file);
  std::lock_guard<std::mutex> lock(n_net_mutex);

  try {
    std::cout << "Force reinitializing neural network..." << std::endl;

    // Reset the loaded flag to allow reinitialization
    n_net_loaded = false;

    const auto t1_start = std::chrono::system_clock::now();

    const std::string current_model = get_current_model_grpc();
    cv::dnn::Net new_net;

    if (current_model != "none") {
      const std::string actual_model_path = get_model_path_for_current_model(
          model_path.substr(0, model_path.find_last_of('/') + 1));
      std::cout << "Loading new model: " << actual_model_path << std::endl;
      new_net = cv::dnn::readNetFromONNX(actual_model_path);
      if (new_net.empty()) {
        // Fallback to default model if selected model fails
        std::cout << "Selected model failed, trying default model..."
                  << std::endl;
        new_net = cv::dnn::readNetFromONNX(model_path);
      }
    } else {
      std::cout << "gRPC not available, using default model..." << std::endl;
      new_net = cv::dnn::readNetFromONNX(model_path);
    }
    if (new_net.empty()) {
      std::cout << "Failed to load any model, keeping current model"
                << std::endl;
      n_net_loaded = true;  // Restore flag since we're keeping current model
      return false;
    }
// Configure a new network
#ifdef USE_CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#else
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
#endif

    // Replace the global network
    net = new_net;
    n_net_loaded = true;

    const auto t1_end = std::chrono::system_clock::now();
    const auto t1_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t1_end - t1_start)
            .count();
    if (enable_profiling)
      spdlog::info("reinitialize_nn_force duration: {} us", t1_duration);

    std::cout << "Neural network reinitialized successfully" << std::endl;
    return true;
  } catch (const cv::Exception& e) {
    std::cerr << "Error during neural network reinitialization: " << e.what()
              << std::endl;
    n_net_loaded = true;  // Restore flag since we failed
    return false;
  }
}

// Function to reinitialize neural network with model switch
bool reinitialize_nn_with_model_switch(const std::string& basePath) {
  try {
    std::cout << "Reinitializing for model switch..." << std::endl;

    // Keep the existing gRPC stream alive - just reload the neural network
    const std::string current_model = get_current_model_grpc();
    std::cout << "Switching to model: " << current_model << std::endl;
    // Force reload the neural network with new model using proper path
    // resolution
    const std::string model_path_for_reinit =
        get_model_path_for_current_model(basePath);
    if (!reinitialize_nn_force(model_path_for_reinit)) {
      std::cout << "Failed to reinitialize neural network" << std::endl;
      return false;
    }
    std::cout << "Model switch completed successfully to: " << current_model
              << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error during model switch: " << e.what() << std::endl;
    return false;
  }
}

cv::dnn::Net& get_network() {
  return net;
}

// Utility function for letterboxing
cv::Mat letterBox(const cv::Mat& image,
                  const cv::Size& newShape,
                  const cv::Scalar& color = cv::Scalar(114, 114, 114)) {
  static int t5_count = 0;
  static long long t5_total_duration = 0;
  const auto t5_start = std::chrono::steady_clock::now();
  // Calculate the scaling ratio to fit the image within the new shape
  float ratio = std::min(
      static_cast<float>(newShape.height) / static_cast<float>(image.rows),
      static_cast<float>(newShape.width) / static_cast<float>(image.cols));

  // Calculate new dimensions after scaling
  const int newUnpadW =
      static_cast<int>(std::round(static_cast<float>(image.cols) * ratio));
  const int newUnpadH =
      static_cast<int>(std::round(static_cast<float>(image.rows) * ratio));

  // Calculate padding needed to reach the desired shape
  const int dw = newShape.width - newUnpadW;
  const int dh = newShape.height - newUnpadH;

  // Evenly distribute padding on both sides
  const int padLeft = dw / 2;
  const int padRight = dw - padLeft;
  const int padTop = dh / 2;
  const int padBottom = dh - padTop;

  cv::Mat resizedImage;
  // Resize the image if the new dimensions differ
  if (image.cols != newUnpadW || image.rows != newUnpadH) {
    cv::resize(image, resizedImage, cv::Size(newUnpadW, newUnpadH), 0, 0,
               cv::INTER_LINEAR);
  } else {
    resizedImage = image;
  }

  cv::Mat paddedImage;
  // Apply padding to reach the desired shape
  cv::copyMakeBorder(resizedImage, paddedImage, padTop, padBottom, padLeft,
                     padRight, cv::BORDER_CONSTANT, color);

  const auto t5_end = std::chrono::steady_clock::now();
  const auto t5_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t5_end - t5_start)
          .count();
  t5_total_duration += t5_duration;
  t5_count++;
  if (t5_count == 10) {
    double t5_average = static_cast<double>(t5_total_duration) / t5_count;
    if (enable_profiling)
      spdlog::info("letterBox {} cycles average duration: {} us", t5_count,
                   t5_average);
    t5_count = 0;
    t5_total_duration = 0;
  }
  return paddedImage;
}

// Utility function to scale coordinates back to the original image
cv::Rect scaleCoords(const cv::Size& imgShape,
                     const cv::Rect& coords,
                     const cv::Size& imgOriginalShape) {
  static int t6_count = 0;
  static long long t6_total_duration = 0;
  const auto t6_start = std::chrono::steady_clock::now();
  const float gain = std::min(static_cast<float>(imgShape.height) /
                                  static_cast<float>(imgOriginalShape.height),
                              static_cast<float>(imgShape.width) /
                                  static_cast<float>(imgOriginalShape.width));

  const int padX = static_cast<int>(
      std::round((static_cast<float>(imgShape.width) -
                  static_cast<float>(imgOriginalShape.width) * gain) /
                 2.0f));
  const int padY = static_cast<int>(
      std::round((static_cast<float>(imgShape.height) -
                  static_cast<float>(imgOriginalShape.height) * gain) /
                 2.0f));

  cv::Rect result;
  result.x =
      static_cast<int>(std::round(static_cast<float>(coords.x - padX) / gain));
  result.y =
      static_cast<int>(std::round(static_cast<float>(coords.y - padY) / gain));
  result.width =
      static_cast<int>(std::round(static_cast<float>(coords.width) / gain));
  result.height =
      static_cast<int>(std::round(static_cast<float>(coords.height) / gain));

  // Clamp coordinates to image boundaries
  result.x = std::max(0, std::min(result.x, imgOriginalShape.width));
  result.y = std::max(0, std::min(result.y, imgOriginalShape.height));
  result.width =
      std::max(0, std::min(result.width, imgOriginalShape.width - result.x));
  result.height =
      std::max(0, std::min(result.height, imgOriginalShape.height - result.y));

  const auto t6_end = std::chrono::steady_clock::now();
  const auto t6_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t6_end - t6_start)
          .count();
  t6_total_duration += t6_duration;
  t6_count++;
  if (t6_count == 10) {
    double t6_average = static_cast<double>(t6_total_duration) / t6_count;
    if (enable_profiling)
      spdlog::info("scaleCoords {} cycles average duration: {} us", t6_count,
                   t6_average);
    t6_count = 0;
    t6_total_duration = 0;
  }
  return result;
}

cv::Mat pwbuffer_to_cvmat(const struct pw_buffer* buf,
                          const uint32_t frame_width,
                          const uint32_t frame_height) {
  static int t2_count = 0;
  static long long t2_total_duration = 0;
  const auto t2_start = std::chrono::steady_clock::now();

  if (!buf || buf->buffer->n_datas == 0)
    return {};
  const struct spa_data* spa_data = &buf->buffer->datas[0];
  auto* data = static_cast<uint8_t*>(spa_data->data);

  const cv::Mat yuy2_frame(frame_height, frame_width, CV_8UC2, data);

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
                       const struct pw_buffer* buf,
                       const uint32_t frame_width,
                       const uint32_t frame_height) {
  if (!buf || visImg.empty())
    return;

  static int t3_count = 0;
  static long long t3_total_duration = 0;
  const auto t3_start = std::chrono::steady_clock::now();

  cv::Mat yuy2_frame;
  cv::resize(visImg, visImg, cv::Size(frame_width, frame_height));
  cv::cvtColor(visImg, yuy2_frame, cv::COLOR_BGR2YUV_YUY2);

  const struct spa_data* spa_data = &buf->buffer->datas[0];
  auto* data = static_cast<uint8_t*>(spa_data->data);

  if (const size_t bytes_to_copy = frame_width * frame_height * 2;
      yuy2_frame.total() * yuy2_frame.elemSize() >= bytes_to_copy &&
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

bool initialize_network_once(const std::string& model_path) {
  static std::once_flag init_flag;
  static bool init_success = false;

  std::call_once(init_flag,
                 [&]() { init_success = initialize_nn(model_path); });

  return init_success;
}

// Thread function that does the actual detection
void detectobjects_worker_thread(AsyncDetectionData* data) {
  bool success = false;

  try {
    // Check if we should stop
    if (g_should_stop) {
      std::cout << "Detection stopped by server request" << std::endl;
      data->callback(data->buffer, data->user_data, false);
      delete data;
      return;
    }
    // Check if we should reinitialize (with mutex to prevent multiple
    // simultaneous switches)
    {
      std::lock_guard<std::mutex> switch_lock(g_switch_mutex);
      if (g_should_reinitialize && !g_reinitializing_in_progress) {
        g_reinitializing_in_progress = true;
        std::cout << "Reinitializing due to model switch..." << std::endl;
        if (reinitialize_nn_with_model_switch(data->basePath)) {
          g_should_reinitialize = false;
          std::cout << "Model switch completed successfully" << std::endl;
        } else {
          std::cout << "Model switch failed, stopping detection" << std::endl;
          g_should_stop = true;
        }
        g_reinitializing_in_progress = false;
      } else if (g_should_reinitialize && g_reinitializing_in_progress) {
        // Another thread is already handling the switch, just return
        std::cout << "Model switch already in progress, skipping this detection"
                  << std::endl;
        data->callback(data->buffer, data->user_data, false);
        delete data;
        return;
      }
    }
    // If stop was triggered during reinitialize, exit
    if (g_should_stop) {
      std::cout << "Detection stopped after model switch" << std::endl;
      data->callback(data->buffer, data->user_data, false);
      delete data;
      return;
    }

    cv::Mat img =
        pwbuffer_to_cvmat(data->buffer, data->frame_width, data->frame_height);

    string yoloModelWeights = get_model_path_for_current_model(data->basePath);

    if (!initialize_network_once(yoloModelWeights)) {
      std::cerr << "Failed to initialize network" << std::endl;
      data->callback(data->buffer, data->user_data, false);
      delete data;
      return;
    }

    // Cache classes loading (load once per thread or globally)
    static thread_local std::vector<string> cached_classes;
    static thread_local std::string cached_classes_file;

    // Get the correct class file based on the current model
    std::string current_class_file =
        get_class_file_for_current_model(data->basePath);

    if (cached_classes_file != current_class_file) {
      cached_classes.clear();
      if (ifstream ifs(current_class_file); ifs.is_open()) {
        string line;
        while (getline(ifs, line)) {
          cached_classes.push_back(line);
        }
        cached_classes_file = current_class_file;
      } else {
        // Fallback to the original class file if the model-specific one doesn't
        // exist
        if (ifstream fallback_ifs(data->classesFile); fallback_ifs.is_open()) {
          string line;
          while (getline(fallback_ifs, line)) {
            cached_classes.push_back(line);
          }
          cached_classes_file = data->classesFile;
        }
      }
    }
    constexpr int inputSize = 640;  // Changed from 640 to match training size
    cv::Size originalSize = img.size();

    // Improved preprocessing with proper letterboxing
    cv::Mat letterboxed = letterBox(img, cv::Size(inputSize, inputSize));

    // Convert to blob with proper normalization
    cv::Mat blob;
    cv::dnn::blobFromImage(letterboxed, blob, 1.0 / 255.0,
                           cv::Size(inputSize, inputSize), cv::Scalar(0, 0, 0),
                           true, false);
    static int t4_count = 0;
    static long long t4_total_duration = 0;
    const auto t4_start = std::chrono::steady_clock::now();
    std::vector<cv::Mat> netOutput;
    {
      std::lock_guard<std::mutex> lock(n_net_mutex);
      net = get_network();
      net.setInput(blob);
      vector<cv::String> names = net.getUnconnectedOutLayersNames();
      net.forward(netOutput, names);
    }
    const auto t4_end = std::chrono::steady_clock::now();
    const auto t4_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t4_end - t4_start)
            .count();
    t4_total_duration += t4_duration;
    t4_count++;
    if (t4_count == 10) {
      double t4_average = static_cast<double>(t4_total_duration) / t4_count;
      if (enable_profiling)
        spdlog::info(
            "Inference core (L211-216) CUDA {} cycles average duration: {} us",
            t4_count, t4_average);
      t4_count = 0;
      t4_total_duration = 0;
    }

    if (netOutput.empty()) {
      cerr << "Error: No network output received" << endl;
      return;
    }

    cv::Mat output = netOutput[0];

    // YOLOv11 output format: [1, num_features, num_detections]
    // Features: [x, y, w, h, class_scores...]

    int dimensions = output.dims;
    if (dimensions != 3) {
      cerr << "Error: Expected 3D output tensor, got " << dimensions << "D"
           << endl;
      return;
    }

    int num_features = output.size[1];    // x, y, w, h + num_classes
    int num_detections = output.size[2];  // Number of detections (e.g., 8400)
    int num_classes = num_features - 4;   // Subtract x, y, w, h
    if (num_classes <= 0) {
      cerr << "Error: Invalid number of classes: " << num_classes << endl;
      return;
    }

    // Get output dimensions
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    auto output_data = reinterpret_cast<const float*>(output.data);

    // Process each detection
    for (int d = 0; d < num_detections; ++d) {
      // Extract bounding box coordinates (center format)
      float centerX = output_data[0 * num_detections + d];
      float centerY = output_data[1 * num_detections + d];
      float width = output_data[2 * num_detections + d];
      float height = output_data[3 * num_detections + d];

      // Find the class with the highest confidence
      int classId = -1;
      float maxScore = -FLT_MAX;
      for (int c = 0; c < num_classes; ++c) {
        if (float score = output_data[(4 + c) * num_detections + d];
            score > maxScore) {
          maxScore = score;
          classId = c;
        }
      }

      // Only keep detections above the confidence threshold
      if (maxScore > data->confThreshold) {
        // Convert from center format to corner format
        float left = centerX - width / 2.0f;
        float top = centerY - height / 2.0f;

        // Create the rectangle in letterboxed image coordinates
        cv::Rect box(static_cast<int>(left), static_cast<int>(top),
                     static_cast<int>(width), static_cast<int>(height));

        // Scale coordinates back to the original image size
        cv::Rect scaledBox =
            scaleCoords(cv::Size(inputSize, inputSize), box, originalSize);

        boxes.push_back(scaledBox);
        classIds.push_back(classId);
        confidences.push_back(maxScore);
      }
    }

    // Perform non-maxima suppression
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, data->confThreshold,
                      data->nmsThreshold, indices);

    std::vector<BoundingBox> bBoxes;
    // Create BoundingBox objects for results and send via gRPC
    for (int idx : indices) {
      BoundingBox bBox;
      bBox.roi = boxes[idx];
      bBox.classID = classIds[idx];
      bBox.confidence = confidences[idx];
      bBox.boxID =
          static_cast<int>(bBoxes.size());  // zero-based unique identifier
      bBoxes.push_back(bBox);

      // Send each detection via gRPC and handle the response
      string class_name = "unknown";
      if (bBox.classID < static_cast<int>(cached_classes.size())) {
        class_name = cached_classes[bBox.classID];
      }

      // Handle gRPC response (only set flags, don't process immediately)
      if (int response = send_detection_grpc(bBox, class_name, bBox.confidence,
                                             bBox.classID);
          response == 1) {  // Switch model
        std::cout << "Received switch command from server" << std::endl;
        std::lock_guard<std::mutex> switch_lock(g_switch_mutex);
        if (!g_should_reinitialize && !g_reinitializing_in_progress) {
          g_should_reinitialize = true;
        }
        break;                     // Stop processing the current frame
      } else if (response == 2) {  // Stop
        std::cout << "Received stop command from server" << std::endl;
        g_should_stop = true;
        break;  // Stop processing the current frame
      }
      // response == 0 means continuing, response == -1 means error but continue
    }

    // Check stream response periodically (only if not already switching)
    {
      std::lock_guard<std::mutex> switch_lock(g_switch_mutex);
      if (!g_should_reinitialize && !g_reinitializing_in_progress) {
        g_detection_count++;
        if (g_detection_count >= CHECK_RESPONSE_INTERVAL) {
          if (int stream_response = check_stream_response_grpc();
              stream_response == 1) {  // Switch
            std::cout << "Stream response: switch model" << std::endl;
            g_should_reinitialize = true;
          } else if (stream_response == 2) {  // Stop
            std::cout << "Stream response: stop detection" << std::endl;
            g_should_stop = true;
          }
          g_detection_count = 0;
        }
      }
    }

    cv::Mat visImg = img.clone();
    // Show results
    if (data->bVis) {
      for (const auto& box : bBoxes) {
        // Draw the rectangle displaying the bounding box
        int top, left, width, height;
        top = box.roi.y;
        left = box.roi.x;
        width = box.roi.width;
        height = box.roi.height;
        cv::rectangle(visImg, cv::Point(left, top),
                      cv::Point(left + width, top + height),
                      cv::Scalar(0, 255, 0), 2);

        string label = cv::format("%.2f", box.confidence);
        if (box.classID < static_cast<int>(cached_classes.size())) {
          label += cached_classes[box.classID] + ":";
        }

        // Display the label at the top of the bounding box
        int baseLine;
        cv::Size labelSize =
            getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        rectangle(
            visImg,
            cv::Point(left,
                      static_cast<int>(top - round(1.5 * labelSize.height))),
            cv::Point(static_cast<int>(left + round(1.5 * labelSize.width)),
                      top + baseLine),
            cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75,
                    cv::Scalar(0, 0, 0), 1);
      }
    }
    cvmat_to_pwbuffer(visImg, data->buffer, data->frame_width,
                      data->frame_height);
    success = true;
  } catch (const std::exception& e) {
    cout << "detection failed: " << e.what() << endl;
    success = false;
  }
  data->callback(data->buffer, data->user_data, success);
  delete data;
}

// Async entry point
void detectObjects_async(struct pw_buffer* out_buffer,
                         const float confThreshold,
                         const float nmsThreshold,
                         const char* basePath,
                         const char* classesFile,
                         const uint32_t frame_width,
                         const uint32_t frame_height,
                         const bool bVis,
                         const detection_callback_t callback,
                         struct impl* user_data) {
  // thread data structure
  auto* data = new AsyncDetectionData{.buffer = out_buffer,
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

void shutdown_detection_with_grpc() {
  cleanup_grpc_client();
}