
#
# CUDA
#
option(WITH_CUDA "Build with CUDA support" OFF)
if(WITH_CUDA)
    set(CUDA_ARCHS "52;61;72;75;86" CACHE STRING "List of architectures to generate device code for")
    enable_language(CUDA)
endif()
