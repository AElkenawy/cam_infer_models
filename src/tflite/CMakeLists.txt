
#
# TensorFlow Lite
#
if (NOT DEFINED TFLITE_ROOT)
    message(FATAL_ERROR "TFLITE_ROOT is not defined. Please set TFLITE_ROOT to the path where TensorFlow Lite is built.")
endif ()

if (NOT DEFINED TFLITE_LIB_PATH)
    message(FATAL_ERROR "TFLITE_LIB_PATH is not defined. Please set TFLITE_LIB_PATH to the path where libtensorflow-lite.so is located.")
endif ()

#
# Shared library for tflite detection
#
add_library(tflite_detection SHARED
        objectDetection_tflite.cc
        ../grpc_client.cc
)

target_include_directories(tflite_detection PRIVATE ${TFLITE_ROOT})

target_link_libraries(tflite_detection
        PUBLIC
        coordinator_grpc_proto
        PkgConfig::OpenCV
        PkgConfig::PIPEWIRE
        spdlog::spdlog
        ${TFLITE_LIB_PATH}/libtensorflowlite.so
)
