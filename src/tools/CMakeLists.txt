include_directories(../../include/stereo_feature_extraction)
rosbuild_add_executable(extractor extractor.cpp)
target_link_libraries(extractor ${PROJECT_NAME})

