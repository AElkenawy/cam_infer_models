
#
# Shared library for YOLO detection
#
add_library(yolo_detection SHARED
        objectDetection2D.cc
        ../grpc_client.cc
)

target_compile_options(yolo_detection PRIVATE -fPIC)

target_link_libraries(yolo_detection
        PUBLIC
        coordinator_grpc_proto
        PkgConfig::OpenCV
        PkgConfig::PIPEWIRE
        spdlog::spdlog
)

if (WITH_CUDA)
    target_link_libraries(yolo_detection CUDA::cudart)
endif ()
