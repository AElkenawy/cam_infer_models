#
# Project Dependencies
#

find_package(PkgConfig REQUIRED)

#
# OpenCV
#
pkg_check_modules(OpenCV IMPORTED_TARGET opencv REQUIRED)

#
# PipeWire and SPA
#
pkg_check_modules(PIPEWIRE IMPORTED_TARGET libpipewire-0.3 libspa-0.2 REQUIRED)

#
# CUDA
#
if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message("${CUDA_LIBRARIES}")
    add_definitions(-DUSE_CUDA)

    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
