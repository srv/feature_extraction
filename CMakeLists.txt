cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()
rosbuild_add_boost_directories()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

rosbuild_add_library(${PROJECT_NAME}
    src/stereo_feature_extraction/stereo_feature_extractor.cpp
    src/stereo_feature_extraction/stereo_camera_model.cpp
    src/stereo_feature_extraction/stereo_feature.cpp
    src/stereo_feature_extraction/drawing.cpp
    src/stereo_feature_extraction/stereo_feature_set.cpp
    src/stereo_feature_extraction/stereo_matching.cpp
    )

rosbuild_add_library(${PROJECT_NAME}_ros
    src/nodelets/stereo_feature_extractor_nodelet.cpp
    src/nodelets/stereo_feature_matcher_nodelet.cpp
    )

target_link_libraries(${PROJECT_NAME}_ros ${PROJECT_NAME})

rosbuild_add_executable(plane_fitting_node src/nodes/plane_fitting_node.cpp)
rosbuild_add_executable(extractor src/tools/extractor.cpp)
target_link_libraries(extractor ${PROJECT_NAME})
rosbuild_link_boost(extractor program_options)
rosbuild_add_gtest(stereo_feature_extractor_test test/stereo_feature_extractor_test.cpp)
target_link_libraries(stereo_feature_extractor_test ${PROJECT_NAME})
 
#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

