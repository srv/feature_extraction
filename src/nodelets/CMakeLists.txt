set(SOURCES
    stereo_feature_extraction_nodelet.cpp
    )
include_directories(../../include/stereo_feature_extraction)
rosbuild_add_library(${PROJECT_NAME} ${SOURCES})
