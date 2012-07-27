#include <gtest/gtest.h>

#include "feature_extraction/features_io.h"

#include "check_equal.h"

using namespace feature_extraction;

std::vector<cv::KeyPoint> randomKeyPoints(int num)
{
  cv::RNG rng;
  std::vector<cv::KeyPoint> key_points(num);
  for (size_t i = 0; i < key_points.size(); ++i)
  {
    key_points[i].pt.x = rng.uniform(0.0, 800.0);
    key_points[i].pt.y = rng.uniform(0.0, 600.0);
    key_points[i].size = rng.uniform(1.0, 100.0);
    key_points[i].response = rng.uniform(1.0, 100.0);
    key_points[i].angle = rng.uniform(0.0, 360.0);
    key_points[i].octave = static_cast<int>(rng.uniform(1, 8));
    key_points[i].class_id = static_cast<int>(rng.uniform(0, 100));
  }
  return key_points;
}

std::vector<cv::Point3d> randomPoints3d(int num)
{
  cv::RNG rng;
  std::vector<cv::Point3d> points(num);
  for (size_t i = 0; i < points.size(); ++i)
  {
    points[i].x = rng.uniform(-5.0, 5.0);
    points[i].y = rng.uniform(-5.0, 5.0);
    points[i].z = rng.uniform(-5.0, 5.0);
  }
  return points;
}

TEST(FeaturesIOTest, iotest)
{
  // some random key points
  static const int NUM_KEY_POINTS = 1000;
  std::vector<cv::KeyPoint> key_points_left = randomKeyPoints(NUM_KEY_POINTS);
  std::vector<cv::KeyPoint> key_points_right = randomKeyPoints(NUM_KEY_POINTS);
  std::vector<cv::Point3d> points3d = randomPoints(NUM_KEY_POINTS);

  cv::Mat descriptors(NUM_KEY_POINTS, 64, CV_32F);
  cv::randu(descriptors, cv::Scalar::all(-128), cv::Scalar::all(128));

  std::string yaml_filename("/tmp/test_features.yaml");
  features_io::saveStereoFeatures(
      yaml_filename, key_points_left, key_points_right, descriptors, points3d);

  std::vector<cv::KeyPoint> loaded_key_points_left;
  std::vector<cv::KeyPoint> loaded_key_points_right;
  cv::Mat loaded_descriptors;
  std::vector<cv::Point3d> loaded_points3d;
  features_io::loadStereoFeatures(
      yaml_filename, loaded_key_points_left, loaded_key_points_right, loaded_descriptors, loaded_points3d);

  checkEqual(key_points_left, loaded_key_points_left);
  checkEqual(key_points_right, loaded_key_points_right);
  checkEqual(points3d, loaded_points3d);
  checkEqual(descriptors, loaded_descriptors);
}

