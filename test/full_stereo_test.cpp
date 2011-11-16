#include <gtest/gtest.h>

#include <opencv2/highgui/highgui.hpp>

#include <ros/package.h>
#include <camera_calibration_parsers/parse.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/stereo_camera_model.h>

#include "feature_matching/stereo_feature_matcher.h"
#include "feature_extraction/feature_extractor_factory.h"

TEST(FullStereo, runTest)
{
  std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
  // load images
  cv::Mat image_left = cv::imread(path + "/data/left_100cm.jpg");
  ASSERT_FALSE(image_left.empty());
  cv::Mat image_right = cv::imread(path + "/data/right_100cm.jpg");
  ASSERT_FALSE(image_right.empty());

  // extract features
  feature_extraction::FeatureExtractor::Ptr extractor = feature_extraction::FeatureExtractorFactory::create("SmartSURF");
  std::vector<feature_extraction::KeyPoint> key_points_l, key_points_r;
  cv::Mat descriptors_left, descriptors_right;
  extractor->extract(image_left, key_points_l, descriptors_left);
  extractor->extract(image_right, key_points_r, descriptors_right);

  std::vector<cv::KeyPoint> key_points_left, key_points_right;
  for (size_t i = 0; i < key_points_l.size(); ++i) key_points_left.push_back(key_points_l[i].toCv());
  for (size_t i = 0; i < key_points_r.size(); ++i) key_points_right.push_back(key_points_r[i].toCv());

  // match features
  feature_matching::StereoFeatureMatcher::Params params;
  params.max_y_diff = 10.0;
  params.max_angle_diff = 10.0;
  params.max_size_diff = 10.0;
  params.min_disparity = 0.0;
  params.max_disparity = 100.0;
  feature_matching::StereoFeatureMatcher matcher;
  matcher.setParams(params);

  double matching_threshold = 0.9;
  std::vector<cv::DMatch> matches;
  matcher.match(key_points_left, descriptors_left, key_points_right, descriptors_right, matching_threshold, matches);

  // read calibration data to fill stereo camera model
  std::string calibration_file_left = path + "/data/real_calibration_left.yaml";
  std::string calibration_file_right = path + "/data/real_calibration_right.yaml";
  sensor_msgs::CameraInfo camera_info_left;
  sensor_msgs::CameraInfo camera_info_right;
  std::string left_name, right_name;
  camera_calibration_parsers::readCalibration(calibration_file_left, left_name, camera_info_left);
  camera_calibration_parsers::readCalibration(calibration_file_right, right_name, camera_info_right);

  image_geometry::StereoCameraModel stereo_camera_model;
  stereo_camera_model.fromCameraInfo(camera_info_left, camera_info_right);

  std::vector<double> distances;
  std::vector<cv::Point3d> world_points(matches.size());
  int num_outliers = 0;
  double outlier_threshold = 0.05; // everything within +-5cm is ok
  double expected_distance = 1.0; // image was shot at 1m distance
  for (size_t i = 0; i < matches.size(); ++i)
  {
    int index_left = matches[i].queryIdx;
    int index_right = matches[i].trainIdx;
    const cv::KeyPoint& key_point_left = key_points_left[index_left];
    const cv::KeyPoint& key_point_right = key_points_right[index_right];
    float disparity = key_point_left.pt.x - key_point_right.pt.x;
    stereo_camera_model.projectDisparityTo3d(key_point_left.pt, disparity, world_points[i]);
    double distance = world_points[i].z;
    if (distance < expected_distance - outlier_threshold || distance > expected_distance + outlier_threshold)
    {
      num_outliers++;
    }
    else
    {
      distances.push_back(world_points[i].z);
    }
  }
  double relative_num_outliers = static_cast<double>(num_outliers) / matches.size();
  std::cout << matches.size() << " matches, " << num_outliers << " outliers (" << relative_num_outliers * 100.0 << "%)" << std::endl;
  EXPECT_LT(relative_num_outliers, 0.2);

  // http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  double m2 = 0; // helper for floating variance computation
  double mean_distance = 0;
  double min_distance = std::numeric_limits<double>::max();
  double max_distance = 0;
  for (size_t i = 0; i < distances.size(); ++i)
  {
      if (distances[i] > max_distance) max_distance = distances[i];
      if (distances[i] < min_distance) min_distance = distances[i];
      double delta = distances[i] - mean_distance;
      mean_distance += delta / (i + 1);
      m2 += delta * (distances[i] - mean_distance);
  }
  double variance = m2 / (distances.size() - 1);
  double stddev = sqrt(variance);
  std::cout << "distances: mean = " << mean_distance
      << " stddev = " << stddev << " min = " << min_distance 
      << " max = " << max_distance << std::endl;
  EXPECT_NEAR(mean_distance, expected_distance, 0.01); // we expect a smaller error than 1cm
  EXPECT_LT(stddev, 0.01);                             // with this std deviation

  cv::Mat canvas;
  cv::drawMatches(image_left, key_points_left, image_right, key_points_right, matches, canvas);
  cv::namedWindow("matches", 0);
  cv::imshow("matches", canvas);
  cv::waitKey(20000);
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

