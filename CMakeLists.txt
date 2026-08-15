cmake_minimum_required(VERSION 3.16)
project(edgeguard VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -O2)

find_package(OpenCV REQUIRED)

# --- core library: everything except main() and tests, so tests can link
# against the exact same compiled code the CLI binary uses ---
add_library(edgeguard_core
  src/geometry.cpp
  src/nms.cpp
  src/letterbox.cpp
  src/labels.cpp
  src/evaluator.cpp
  src/detector.cpp
)
target_include_directories(edgeguard_core PUBLIC include)
target_link_libraries(edgeguard_core PUBLIC ${OpenCV_LIBS})
target_include_directories(edgeguard_core PUBLIC ${OpenCV_INCLUDE_DIRS})

# Generates include/edgeguard/test_config.hpp (build-dir copy) with an
# absolute EDGEGUARD_SOURCE_DIR path, so tests can reliably locate fixture
# files under models/ regardless of the build directory's location or
# the working directory tests are invoked from.
configure_file(
  ${CMAKE_SOURCE_DIR}/include/edgeguard/test_config.hpp.in
  ${CMAKE_BINARY_DIR}/generated/edgeguard/test_config.hpp
  @ONLY
)

# --- CLI application ---
add_executable(edgeguard src/main.cpp)
target_link_libraries(edgeguard PRIVATE edgeguard_core)

# --- benchmark harness ---
add_executable(edgeguard_bench src/bench.cpp)
target_link_libraries(edgeguard_bench PRIVATE edgeguard_core)

# --- evaluation harness (mAP over a full validation split) ---
add_executable(edgeguard_eval src/eval.cpp)
target_link_libraries(edgeguard_eval PRIVATE edgeguard_core)

# --- tests ---
enable_testing()
find_package(Catch2 3 REQUIRED)

add_executable(edgeguard_tests
  tests/test_geometry.cpp
  tests/test_nms.cpp
  tests/test_letterbox.cpp
  tests/test_labels.cpp
  tests/test_evaluator.cpp
  tests/test_detector_decode.cpp
  tests/test_detector_integration.cpp
)
target_link_libraries(edgeguard_tests PRIVATE edgeguard_core Catch2::Catch2WithMain)
target_include_directories(edgeguard_tests PRIVATE ${CMAKE_BINARY_DIR}/generated)

include(CTest)
add_test(NAME edgeguard_tests COMMAND edgeguard_tests)
