set(SOURCES
    plane_fitting_node.cpp)
include_directories(../../include/stereo_feature_extraction)
rosbuild_add_executable(plane_fitting_node ${SOURCES})
target_link_libraries(plane_fitting_node ${PROJECT_NAME})
