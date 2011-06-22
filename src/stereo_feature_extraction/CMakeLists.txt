set(SOURCES
    feature_extractor_factory.cpp
    stereo_feature_extractor.cpp
    cv_surf_extractor.cpp
    surf_extractor.cpp
    stereo_camera_model.cpp
    stereo_feature.cpp
    Surf.cpp
    drawing.cpp
    ../nodelets/stereo_feature_extractor_nodelet.cpp)
include_directories(../../include/stereo_feature_extraction)
rosbuild_add_library(${PROJECT_NAME} ${SOURCES})
