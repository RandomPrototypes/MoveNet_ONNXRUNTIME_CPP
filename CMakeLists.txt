cmake_minimum_required(VERSION 3.11.0)
project(MoveNet_ONNXRUNTIME_CPP VERSION 0.1 LANGUAGES CXX)

include(FetchContent)

include(CTest)
enable_testing()


find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

find_package(OpenCV REQUIRED)

include_directories(${ONNX_RUNTIME_SESSION_INCLUDE_DIRS})

add_executable(MoveNet_ONNXRUNTIME_CPP demo_singlepose_onnx.cpp)

target_link_libraries(MoveNet_ONNXRUNTIME_CPP PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)
