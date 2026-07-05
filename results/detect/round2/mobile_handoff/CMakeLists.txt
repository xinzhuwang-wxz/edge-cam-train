# 桌面端独立 demo 构建（验证 C++ 核心用；移动端集成见 README §4）。
# 前置：ncnn 与 OpenCV 已安装（或设 ncnn_DIR / OpenCV_DIR 指向各自的 cmake 目录）。
#   mkdir build && cd build
#   cmake .. -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
#   cmake --build . -j
#   ./feeder_demo ../demo_bird.jpg 0.4 0.5 out.jpg   # 需把 .param/.bin 拷到运行目录
cmake_minimum_required(VERSION 3.10)
project(feeder_demo CXX)
set(CMAKE_CXX_STANDARD 11)

find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(feeder_demo demo_main.cpp feeder_detector.cpp)
target_include_directories(feeder_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(feeder_demo PRIVATE ncnn ${OpenCV_LIBS})
