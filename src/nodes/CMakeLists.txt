include_directories(../../include/stereo_feature_extraction)

rosbuild_add_executable(plane_fitting_node plane_fitting_node.cpp)
rosbuild_add_executable(feature_extractor_node feature_extractor_node.cpp)

target_link_libraries(plane_fitting_node ${PROJECT_NAME})
target_link_libraries(feature_extractor_node ${PROJECT_NAME})
