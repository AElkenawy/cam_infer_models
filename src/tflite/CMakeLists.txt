
#
# TensorFlow Lite
#
if (NOT DEFINED TFLITE_ROOT)
    message(FATAL_ERROR "TFLITE_ROOT is not defined. Please set TFLITE_ROOT to the path where TensorFlow Lite is built.")
endif ()

if (NOT DEFINED TFLITE_LIBRARY)
    message(FATAL_ERROR "TFLITE_LIBRARY is not defined. Please set TFLITE_LIBRARY to the absolute path of libtensorflow-lite.so.")
endif ()

#
# tflite detection library
#
add_library(tflite_detection STATIC
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
        ${TFLITE_LIBRARY} -v
)
