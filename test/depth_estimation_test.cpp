#include <fstream>
#include <sstream>
#include <gtest/gtest.h>

#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <ros/package.h>
#include <camera_calibration_parsers/parse.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/stereo_camera_model.h>

#include "feature_matching/stereo_feature_matcher.h"
#include "feature_extraction/key_point_detector_factory.h"
#include "feature_extraction/descriptor_extractor_factory.h"

struct PlaneView
{
  std::string image_left_file;
  std::string image_right_file;
  double expected_distance;
};

struct Camera
{
  std::string name;
  std::string calibration_file_left;
  std::string calibration_file_right;
};

struct PlaneTestSet
{
  Camera camera;
  std::vector<PlaneView> plane_views;
};

void findTestSets(std::vector<PlaneTestSet>& test_sets)
{
  test_sets.clear();

  namespace bf = boost::filesystem;
  namespace ba = boost::algorithm;
  bf::path data_path(ros::package::getPath(ROS_PACKAGE_NAME) + "/data/");
  for (bf::directory_iterator iter(data_path); iter != bf::directory_iterator(); ++iter)
  {
    if (bf::is_directory(iter->path()) && ba::starts_with(iter->path().filename().string(), "camera"))
    {
      PlaneTestSet plane_test_set;
      plane_test_set.camera.name = iter->path().filename().string();
      std::cout << "Collecting images for " << plane_test_set.camera.name << "..." << std::endl;
      bf::path calib_left_path(iter->path() / "calibration_left.yaml");
      bf::path calib_right_path(iter->path() / "calibration_right.yaml");

      ASSERT_TRUE(bf::is_regular_file(calib_left_path)) << " left calibration file not found!";
      ASSERT_TRUE(bf::is_regular_file(calib_right_path)) << " right calibration file not found!";

      plane_test_set.camera.calibration_file_left = calib_left_path.string();
      plane_test_set.camera.calibration_file_right = calib_right_path.string();

      std::string truth_file = (iter->path() / "truth.txt").string();

      std::ifstream in(truth_file.c_str());
      ASSERT_TRUE(in.is_open()) << " cannot open truth file : " << truth_file;

      std::string line;
      while (in.good())
      {
        std::getline(in, line);
        if (line.size() == 0 || line[0] == '#') continue;
        std::istringstream istr(line);
        std::string left_image_file;
        std::string right_image_file;
        double expected_distance;
        istr >> left_image_file;
        istr >> right_image_file;
        istr >> expected_distance;

        std::cout << left_image_file << " " << right_image_file << " " << expected_distance << std::endl;

        PlaneView plane_view;
        plane_view.image_left_file = (iter->path() / left_image_file).string();
        plane_view.image_right_file = (iter->path() / right_image_file).string();
        plane_view.expected_distance = expected_distance;

        plane_test_set.plane_views.push_back(plane_view);
      }
      test_sets.push_back(plane_test_set);
    }
  }
}

TEST(DepthEstimation, runTest)
{
  std::vector<PlaneTestSet> test_sets;
  findTestSets(test_sets);

  feature_extraction::KeyPointDetector::Ptr detector =
    feature_extraction::KeyPointDetectorFactory::create("SmartSURF");
  feature_extraction::DescriptorExtractor::Ptr extractor = 
    feature_extraction::DescriptorExtractorFactory::create("SmartSURF");

  for (size_t i = 0; i < test_sets.size(); ++i)
  {
    // read camera info
    sensor_msgs::CameraInfo camera_info_left;
    sensor_msgs::CameraInfo camera_info_right;
    std::string left_name, right_name;
    camera_calibration_parsers::readCalibration(test_sets[i].camera.calibration_file_left, left_name, camera_info_left);
    camera_calibration_parsers::readCalibration(test_sets[i].camera.calibration_file_right, right_name, camera_info_right);

    image_geometry::StereoCameraModel stereo_camera_model;
    stereo_camera_model.fromCameraInfo(camera_info_left, camera_info_right);

    std::cout << "*** TESTS FOR " << test_sets[i].camera.name << " ***" << std::endl;

    for (size_t j = 0; j < test_sets[i].plane_views.size(); ++j)
    {
      const PlaneView& plane_view = test_sets[i].plane_views[j];
      std::vector<cv::KeyPoint> key_points_left, key_points_right;
      cv::Mat descriptors_left, descriptors_right;
      cv::Mat image_left = cv::imread(plane_view.image_left_file, 0);
      cv::Mat image_right = cv::imread(plane_view.image_right_file, 0);
      detector->detect(image_left, key_points_left);
      detector->detect(image_right, key_points_right);
      extractor->extract(image_left, key_points_left, descriptors_left);
      extractor->extract(image_right, key_points_right, descriptors_right);
      std::cout << key_points_left.size() << " key points in left image, " << 
        key_points_right.size() << " key points in right image." << std::endl;

      EXPECT_EQ(key_points_left.size(), descriptors_left.rows);
      EXPECT_EQ(key_points_right.size(), descriptors_right.rows);

      // match features
      feature_matching::StereoFeatureMatcher::Params params;
      params.max_y_diff = 10.0;
      params.max_angle_diff = 10.0;
      params.max_size_diff = 10.0;
      feature_matching::StereoFeatureMatcher matcher;
      matcher.setParams(params);

      double matching_threshold = 0.9;
      std::vector<cv::DMatch> matches;
      matcher.match(key_points_left, descriptors_left, key_points_right, 
          descriptors_right, matching_threshold, matches);

      EXPECT_GT(matches.size(), 0);

      std::vector<double> distances;
      std::vector<cv::Point3d> world_points(matches.size());
      int num_outliers = 0;
      double expected_distance = plane_view.expected_distance;
      double outlier_threshold = expected_distance / 20; // everything within 5% is ok
      for (size_t k = 0; k < matches.size(); ++k)
      {
        int index_left = matches[k].queryIdx;
        int index_right = matches[k].trainIdx;
        const cv::KeyPoint& key_point_left = key_points_left[index_left];
        const cv::KeyPoint& key_point_right = key_points_right[index_right];
        float disparity = key_point_left.pt.x - key_point_right.pt.x;
        stereo_camera_model.projectDisparityTo3d(key_point_left.pt, disparity, world_points[k]);
        double distance = world_points[k].z;
        if (distance < expected_distance - outlier_threshold || 
            distance > expected_distance + outlier_threshold)
        {
          num_outliers++;
        }
        else
        {
          distances.push_back(world_points[k].z);
        }
      }
      double relative_num_outliers = static_cast<double>(num_outliers) / matches.size();
      std::cout << matches.size() << " matches, " << num_outliers << " outliers (" << relative_num_outliers * 100.0 << "%)" << std::endl;
      EXPECT_LT(relative_num_outliers, 0.3);

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
      // expected distance / stddev check for inliers
      EXPECT_NEAR(mean_distance, expected_distance, expected_distance/100);
      EXPECT_LT(stddev, expected_distance/50);

      cv::Mat canvas;
      cv::drawMatches(image_left, key_points_left, image_right, key_points_right, matches, canvas);
      cv::namedWindow("matches", 0);
      cv::imshow("matches", canvas);
      cv::waitKey(500);
    }
  }
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

