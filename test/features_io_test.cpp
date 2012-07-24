#include <gtest/gtest.h>

#include "feature_extraction/features_io.h"

using namespace feature_extraction;

TEST(FeaturesIOTest, iotest)
{
  // some random key points
  cv::RNG rng;
  static const int NUM_KEY_POINTS = 1000;
  std::vector<cv::KeyPoint> key_points(NUM_KEY_POINTS);
  std::vector<cv::Point3d> world_points(NUM_KEY_POINTS);
  for (size_t i = 0; i < key_points.size(); ++i)
  {
    key_points[i].pt.x = rng.uniform(0.0, 800.0);
    key_points[i].pt.y = rng.uniform(0.0, 600.0);
    key_points[i].size = rng.uniform(1.0, 100.0);
    key_points[i].response = rng.uniform(1.0, 100.0);
    key_points[i].angle = rng.uniform(0.0, 360.0);
    key_points[i].octave = static_cast<int>(rng.uniform(1, 8));
    key_points[i].class_id = static_cast<int>(rng.uniform(0, 100));
    world_points[i].x = rng.uniform(-5.0, 5.0);
    world_points[i].y = rng.uniform(-5.0, 5.0);
    world_points[i].z = rng.uniform(-5.0, 5.0);
  }

  cv::Mat descriptors(NUM_KEY_POINTS, 64, CV_32F);
  cv::randu(descriptors, cv::Scalar::all(-128), cv::Scalar::all(128));


  std::string yaml_filename("/tmp/test_features.yaml");
  features_io::saveStereoFeatures(
      yaml_filename, key_points, descriptors, world_points);

  std::vector<cv::KeyPoint> loaded_key_points;
  cv::Mat loaded_descriptors;
  std::vector<cv::Point3d> loaded_world_points;
  features_io::loadStereoFeatures(
      yaml_filename, loaded_key_points, loaded_descriptors, loaded_world_points);
}

