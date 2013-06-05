cmake_minimum_required(VERSION 2.6)
find_package(OpenCV REQUIRED)

include_directories(../include)
include_directories(/usr/include/opencv2)

link_directories(/usr/local/lib)
link_directories(/usr/lib)

add_executable(objectTracking objectTracking.cpp)
target_link_libraries(objectTracking ${OpenCV_LIBS})

add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS})

add_executable(cameraCalibration cameraCalibration.cpp)
target_link_libraries(cameraCalibration ${OpenCV_LIBS})
