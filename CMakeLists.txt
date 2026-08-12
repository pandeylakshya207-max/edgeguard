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

# --- CLI application ---
add_executable(edgeguard src/main.cpp)
target_link_libraries(edgeguard PRIVATE edgeguard_core)

# --- benchmark harness ---
add_executable(edgeguard_bench src/bench.cpp)
target_link_libraries(edgeguard_bench PRIVATE edgeguard_core)

# --- tests ---
enable_testing()
find_package(Catch2 3 REQUIRED)

add_executable(edgeguard_tests
  tests/test_geometry.cpp
  tests/test_nms.cpp
  tests/test_letterbox.cpp
  tests/test_labels.cpp
  tests/test_evaluator.cpp
)
target_link_libraries(edgeguard_tests PRIVATE edgeguard_core Catch2::Catch2WithMain)

include(CTest)
add_test(NAME edgeguard_tests COMMAND edgeguard_tests)
