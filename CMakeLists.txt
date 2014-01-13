cmake_minimum_required(VERSION 2.8.3)
project(feature_extraction)

find_package(catkin REQUIRED COMPONENTS 
	roscpp
	rospy
	pcl_ros 
	cv_bridge 
	image_transport 
	image_geometry
	vision_msgs
	camera_calibration_parsers)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED COMPONENTS program_options system filesystem)

catkin_package(INCLUDE_DIRS include
               LIBRARIES ${PROJECT_NAME} feature_matching)

include_directories(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} include)

# LIBRARIES
add_library(${PROJECT_NAME}
	src/feature_extraction/key_point_detector_factory.cpp
	src/feature_extraction/descriptor_extractor_factory.cpp
	src/feature_extraction/cv_key_point_detector.cpp
	src/feature_extraction/cv_descriptor_extractor.cpp
	src/feature_extraction/smart_surf.cpp
	src/feature_extraction/smart_surf_key_point_detector.cpp
	src/feature_extraction/smart_surf_descriptor_extractor.cpp
	src/feature_extraction/key_points_filter.cpp
	src/feature_extraction/features_io.cpp
	)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_library(feature_matching
	src/feature_matching/matching_methods.cpp
	src/feature_matching/stereo_feature_matcher.cpp
	src/feature_matching/stereo_depth_estimator.cpp
	src/feature_matching/estimate_rigid_transformation.cpp
	)
target_link_libraries(feature_matching ${Boost_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

# EXECUTABLES
add_executable(stereo_extractor src/tools/stereo_extractor.cpp)
target_link_libraries(stereo_extractor ${PROJECT_NAME} feature_matching ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(feature_extractor src/tools/feature_extractor.cpp)
target_link_libraries(feature_extractor ${PROJECT_NAME} feature_matching ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(feature3d_matcher src/tools/feature3d_matcher.cpp)
target_link_libraries(feature3d_matcher ${PROJECT_NAME} feature_matching ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(stereo_matcher src/tools/stereo_matcher.cpp)
target_link_libraries(stereo_matcher ${PROJECT_NAME} feature_matching ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

# UNIT TESTS
catkin_add_gtest(key_point_detectors_test test/key_point_detectors_test.cpp)
target_link_libraries(key_point_detectors_test ${PROJECT_NAME} gtest_main)

catkin_add_gtest(descriptor_extractors_test test/descriptor_extractors_test.cpp)
target_link_libraries(descriptor_extractors_test ${PROJECT_NAME} gtest_main)

catkin_add_gtest(detector_extractor_combinations_test test/detector_extractor_combinations_test.cpp)
target_link_libraries(detector_extractor_combinations_test ${PROJECT_NAME} gtest_main)

catkin_add_gtest(key_points_filter_test test/key_points_filter_test.cpp)
target_link_libraries(key_points_filter_test ${PROJECT_NAME} gtest_main)

catkin_add_gtest(stereo_feature_matcher_test test/stereo_feature_matcher_test.cpp)
target_link_libraries(stereo_feature_matcher_test feature_matching)

catkin_add_gtest(full_stereo_test test/full_stereo_test.cpp)
target_link_libraries(full_stereo_test ${PROJECT_NAME} feature_matching)

catkin_add_gtest(matching_methods_test test/matching_methods_test.cpp)
target_link_libraries(matching_methods_test feature_matching)

catkin_add_gtest(features_io_test test/features_io_test.cpp)
target_link_libraries(features_io_test ${PROJECT_NAME} gtest_main)

catkin_add_gtest(depth_estimation_test test/depth_estimation_test.cpp)
target_link_libraries(depth_estimation_test ${PROJECT_NAME} feature_matching ${Boost_LIBRARIES})
