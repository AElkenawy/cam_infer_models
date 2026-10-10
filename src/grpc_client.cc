#include <grpcpp/grpcpp.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "coordinator.grpc.pb.h"
#include "dataStructures.h"

using coordinator::Ack;
using coordinator::BoundBox;
using coordinator::ConnectRequest;
using coordinator::ConnectResponse;
using coordinator::Coordinator;
using coordinator::Detection;
using coordinator::Model;
using coordinator::ModelListRequest;
using coordinator::ModelListResponse;
using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReaderWriter;
using grpc::Status;

class InferenceGrpcClient {
 public:
  explicit InferenceGrpcClient(const std::shared_ptr<Channel>& channel)
      : stub_(Coordinator::NewStub(channel)),
        control_id_(0),
        connected_(false),
        streaming_active_(false),
        should_stop_reader_(false),
        last_server_response_(0) {}

  ~InferenceGrpcClient() { StopDetectionStream(); }

  bool RegisterModelsAndGetSelected() {
    // Step 1: Send available models to the server
    ModelListRequest request;

    // Add available YOLO models (names must match
    // get_model_path_for_current_model())
    auto* model1 = request.add_available_models();
    model1->set_model_name("traffic_signs");
    model1->set_model_version("Yolo11");

    auto* model2 = request.add_available_models();
    model2->set_model_name("security_mode");
    model2->set_model_version("Yolo11");

    ModelListResponse response;
    ClientContext context;

    if (const Status status =
            stub_->registerModels(&context, request, &response);
        status.ok() && response.success()) {
      control_id_ = response.control_id();
      current_model_name_ = response.selected_model().model_name();
      current_model_version_ = response.selected_model().model_version();

      std::cout << "Models registered. Selected model: " << current_model_name_
                << " v" << current_model_version_
                << " (control_id: " << control_id_ << ")" << std::endl;
      return true;
    } else {
      std::cout << "Model registration failed: " << status.error_message()
                << std::endl;
      return false;
    }
  }

  bool ConnectWithSelectedModel() {
    if (control_id_ == 0 || current_model_name_.empty()) {
      std::cout << "No model selected. Register models first." << std::endl;
      return false;
    }

    // Step 3: Connect with the selected model
    ConnectRequest request;
    request.set_control_id(control_id_);

    auto* selected_model = request.mutable_selected_model();
    selected_model->set_model_name(current_model_name_);
    selected_model->set_model_version(current_model_version_);

    ConnectResponse response;
    ClientContext context;

    if (const Status status = stub_->connect(&context, request, &response);
        status.ok() && response.accepted()) {
      connected_ = true;
      std::cout << "Connected successfully with model: " << current_model_name_
                << std::endl;
      return true;
    }
    std::cout << "Connection failed: " << response.error_msg() << std::endl;
    return false;
  }

  bool StartDetectionStream() {
    if (!connected_ || streaming_active_) {
      return false;
    }

    detection_context_ = std::make_unique<ClientContext>();
    detection_stream_ = stub_->streamDetections(detection_context_.get());

    if (!detection_stream_) {
      std::cout << "Failed to create detection stream" << std::endl;
      return false;
    }

    streaming_active_ = true;
    should_stop_reader_ = false;
    last_server_response_ = 0;

    // Start a background thread to read server responses
    response_reader_thread_ =
        std::thread(&InferenceGrpcClient::ResponseReaderLoop, this);

    std::cout << "Detection streaming started" << std::endl;
    return true;
  }

  // Background thread function to continuously read server responses
  void ResponseReaderLoop() {
    Ack response;
    while (!should_stop_reader_ && streaming_active_) {
      if (detection_stream_->Read(&response)) {
        if (response.success()) {
          last_server_response_ = 0;  // Continue
        } else {
          if (const std::string& message = response.message();
              message.substr(0, 6) == "switch") {
            last_server_response_ = 1;  // Switch

            // Parse the target model from the message (format:
            // "switch:model_name")
            if (const size_t colon_pos = message.find(':');
                colon_pos != std::string::npos &&
                colon_pos + 1 < message.length()) {
              std::string target_model = message.substr(colon_pos + 1);
              current_model_name_ = target_model;
              current_model_version_ = "Yolo11";  // Keep same version
              std::cout << "Server response: switch to model " << target_model
                        << std::endl;
            } else {
              std::cout << "Server response: switch model (no target specified)"
                        << std::endl;
            }
          } else if (message == "stop") {
            last_server_response_ = 2;  // Stop
            std::cout << "Server response: stop detection" << std::endl;
            break;
          }
        }
      } else {
        // Stream ended or error
        std::cout << "Detection stream read ended" << std::endl;
        break;
      }
    }
  }

  // Returns: 0=continue, 1=switch, 2=stop, -1=error
  [[nodiscard]] int SendDetection(const BoundingBox& bbox,
                                  const std::string& label,
                                  const float confidence,
                                  const int class_id) const {
    if (!connected_ || !streaming_active_ || !detection_stream_) {
      return -1;
    }

    Detection detection;
    detection.set_class_(label);
    detection.set_confidence(confidence);
    detection.set_class_id(class_id);
    detection.set_ts_micros(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());

    auto* bbox_proto = detection.mutable_bbox();
    bbox_proto->set_x(bbox.roi.x);
    bbox_proto->set_y(bbox.roi.y);
    bbox_proto->set_width(bbox.roi.width);
    bbox_proto->set_height(bbox.roi.height);

    // Send the detection
    if (!detection_stream_->Write(detection)) {
      std::cout << "Failed to write detection" << std::endl;
      return -1;
    }

    std::cout << "Detection sent: " << label << " (" << confidence << ") at ["
              << bbox.roi.x << "," << bbox.roi.y << "," << bbox.roi.width << ","
              << bbox.roi.height << "]" << std::endl;

    // Return the latest server response
    return last_server_response_.load();
  }

  // Get the latest server response without sending anything
  [[nodiscard]] int GetLatestServerResponse() const {
    return last_server_response_.load();
  }

  void StopDetectionStream() {
    if (streaming_active_) {
      should_stop_reader_ = true;

      if (detection_stream_) {
        detection_stream_->WritesDone();
      }

      if (response_reader_thread_.joinable()) {
        response_reader_thread_.join();
      }

      if (detection_stream_) {
        if (const Status status = detection_stream_->Finish(); !status.ok()) {
          std::cout << "Stream finished with error: " << status.error_message()
                    << std::endl;
        }
      }

      streaming_active_ = false;
      std::cout << "Detection streaming stopped" << std::endl;
    }
  }

  [[nodiscard]] bool IsConnected() const { return connected_; }

  [[nodiscard]] bool IsStreaming() const { return streaming_active_; }

  [[nodiscard]] std::string GetCurrentModel() const {
    return current_model_name_;
  }

  [[nodiscard]] uint32_t GetControlId() const { return control_id_; }

  // Reset for model switching
  void Reset() {
    std::cout << "Resetting gRPC client for model switch..." << std::endl;
    StopDetectionStream();
    connected_ = false;
    control_id_ = 0;
    current_model_name_.clear();
    current_model_version_.clear();
    std::cout << "gRPC client reset completed" << std::endl;
  }

 private:
  std::unique_ptr<Coordinator::Stub> stub_;
  uint32_t control_id_;
  bool connected_;
  std::string current_model_name_;
  std::string current_model_version_;

  // Bidirectional streaming detection members
  std::unique_ptr<ClientReaderWriter<Detection, Ack>> detection_stream_;
  std::unique_ptr<ClientContext> detection_context_;
  bool streaming_active_;

  // Background thread for reading server responses
  std::thread response_reader_thread_;
  std::atomic<bool> should_stop_reader_;
  std::atomic<int> last_server_response_;  // 0=continue, 1=switch, 2=stop
};

// Global instance
static std::unique_ptr<InferenceGrpcClient> g_grpc_client;

// Initialize gRPC client and register models
bool init_grpc_client(const std::string& server_address = "localhost:50051") {
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  g_grpc_client = std::make_unique<InferenceGrpcClient>(channel);

  // Step 1 & 2: Register models and get selected model
  if (!g_grpc_client->RegisterModelsAndGetSelected()) {
    std::cout << "Failed to register models" << std::endl;
    return false;
  }

  // Step 3: Connect with the selected model
  if (!g_grpc_client->ConnectWithSelectedModel()) {
    std::cout << "Failed to connect with selected model" << std::endl;
    return false;
  }

  // Step 4: Start the detection stream
  if (!g_grpc_client->StartDetectionStream()) {
    std::cout << "Failed to start detection stream" << std::endl;
    return false;
  }

  std::cout << "gRPC client initialized successfully with model: "
            << g_grpc_client->GetCurrentModel() << std::endl;
  return true;
}

// Send detection via gRPC and return control message
// Returns: 0=continue, 1=switch, 2=stop, -1=error
int send_detection_grpc(const BoundingBox& bbox,
                        const std::string& label,
                        const float confidence,
                        const int class_id) {
  if (g_grpc_client && g_grpc_client->IsConnected()) {
    return g_grpc_client->SendDetection(bbox, label, confidence, class_id);
  }
  return -1;
}

// Check latest server response (no longer closes stream)
// Returns: 0=continue, 1=switch, 2=stop, -1=error
int check_stream_response_grpc() {
  if (g_grpc_client && g_grpc_client->IsStreaming()) {
    return g_grpc_client->GetLatestServerResponse();
  }
  return -1;
}

// Get current model info
std::string get_current_model_grpc() {
  if (g_grpc_client) {
    return g_grpc_client->GetCurrentModel();
  }
  return "none";
}

// Reset for model switching
void reset_grpc_client() {
  if (g_grpc_client) {
    g_grpc_client->Reset();
  }
}

// Cleanup
void cleanup_grpc_client() {
  if (g_grpc_client) {
    std::cout << "Cleaning up gRPC client..." << std::endl;
    g_grpc_client->StopDetectionStream();
    g_grpc_client.reset();
  }
}