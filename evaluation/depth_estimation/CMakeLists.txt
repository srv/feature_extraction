set(SOURCES
    plane_fitting.cpp)
include_directories(../../include/stereo_feature_extraction)
rosbuild_add_executable(plane_fitting ${SOURCES})
target_link_libraries(plane_fitting ${PROJECT_NAME})
