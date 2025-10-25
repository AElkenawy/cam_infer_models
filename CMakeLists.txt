cmake_minimum_required(VERSION 3.10.2)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug, Release, RelWithDebInfo, or MinSizeRel." FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set, defaulting to Release.")
endif ()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/third_party/sanitizers-cmake/cmake)

if (NOT BUILD_NUMBER)
    set(BUILD_NUMBER 0)
endif ()

project(cam-infer-models
        VERSION "0.1.${BUILD_NUMBER}"
        DESCRIPTION "Camera Inference Models with YOLO and TensorFlow Lite"
        LANGUAGES CXX C
)

include(compiler)
include(options)
include(deps)

add_subdirectory(third_party)
add_subdirectory(camera_idl)

include_directories(include)

add_subdirectory(src/yolo)
add_subdirectory(src/tflite)

#
# YOLO
#

add_executable(filter_g.yolo src/filter_g.cc)

target_link_libraries(filter_g.yolo PUBLIC yolo_detection)

if (WITH_CUDA)
    target_link_libraries(filter_g.yolo CUDA::cudart)
endif ()

install(TARGETS filter_g.yolo DESTINATION bin)


#
# Tensorflow Lite
#

add_executable(filter_g.tflite src/filter_g.cc)

target_link_libraries(filter_g.tflite PUBLIC tflite_detection)

install(TARGETS filter_g.tflite DESTINATION bin)