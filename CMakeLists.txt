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

# LIBRARIES
rosbuild_add_library(${PROJECT_NAME} 
    src/feature_extraction/key_point_detector_factory.cpp
    src/feature_extraction/descriptor_extractor_factory.cpp
    src/feature_extraction/cv_key_point_detector.cpp
    src/feature_extraction/cv_descriptor_extractor.cpp
    src/feature_extraction/smart_surf.cpp
    src/feature_extraction/smart_surf_key_point_detector.cpp
    src/feature_extraction/smart_surf_descriptor_extractor.cpp
    src/feature_extraction/key_points_filter.cpp
    )

rosbuild_add_library(feature_matching
    src/feature_matching/matching_methods.cpp
    src/feature_matching/stereo_feature_matcher.cpp
    src/feature_matching/stereo_depth_estimator.cpp
    )

rosbuild_add_library(nodelets
    src/nodelets/feature_extractor_nodelet.cpp
    src/nodelets/stereo_feature_matcher_nodelet.cpp
    )

target_link_libraries(nodelets ${PROJECT_NAME} feature_matching)


# EXECUTABLES
rosbuild_add_executable(plane_fitting_node src/nodes/plane_fitting_node.cpp)
#rosbuild_add_executable(extractor src/tools/extractor.cpp)
#target_link_libraries(extractor ${PROJECT_NAME})
#rosbuild_link_boost(extractor program_options)

# UNIT TESTS
rosbuild_add_gtest(key_point_detectors_test test/key_point_detectors_test.cpp)
target_link_libraries(key_point_detectors_test ${PROJECT_NAME} gtest_main)

rosbuild_add_gtest(descriptor_extractors_test test/descriptor_extractors_test.cpp)
target_link_libraries(descriptor_extractors_test ${PROJECT_NAME} gtest_main)

rosbuild_add_gtest(key_points_filter_test test/key_points_filter_test.cpp)
target_link_libraries(key_points_filter_test ${PROJECT_NAME} gtest_main)

rosbuild_add_gtest(conversions_test test/conversions_test.cpp)
  
rosbuild_add_gtest(stereo_feature_matcher_test test/stereo_feature_matcher_test.cpp)
target_link_libraries(stereo_feature_matcher_test feature_matching)

rosbuild_add_gtest(full_stereo_test test/full_stereo_test.cpp)
target_link_libraries(full_stereo_test ${PROJECT_NAME} feature_matching)

rosbuild_add_gtest(matching_methods_test test/matching_methods_test.cpp)
target_link_libraries(matching_methods_test feature_matching)

rosbuild_add_gtest(depth_estimation_test test/depth_estimation_test.cpp)
target_link_libraries(depth_estimation_test ${PROJECT_NAME} feature_matching)
rosbuild_link_boost(depth_estimation_test system filesystem)


