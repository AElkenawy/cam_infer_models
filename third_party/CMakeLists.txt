
#
# Speedlog
#
set(SPDLOG_NO_EXCEPTIONS ON)
set(SPDLOG_NO_THREAD_ID ON)
set(SPDLOG_BUILD_PIC ON)
add_library(spdlog INTERFACE)
target_compile_options(spdlog INTERFACE -isystem${CMAKE_CURRENT_SOURCE_DIR}/spdlog/include)
add_library(spdlog::spdlog ALIAS spdlog)
