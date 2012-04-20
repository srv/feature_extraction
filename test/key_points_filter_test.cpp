#include <gtest/gtest.h>
#include <opencv2/features2d/features2d.hpp>

#include "feature_extraction/key_points_filter.h"

using namespace feature_extraction;

TEST(KeyPointsFilterTest, filterBestTest)
{
  // some random key points
  cv::RNG rng;
  static const int NUM_KEY_POINTS = 1000;
  std::vector<cv::KeyPoint> key_points(NUM_KEY_POINTS);
  for (size_t i = 0; i < key_points.size(); ++i)
  {
    key_points[i].pt.x = rng.uniform(0.0, 800.0);
    key_points[i].pt.y = rng.uniform(0.0, 600.0);
    key_points[i].size = rng.uniform(1.0, 100.0);
    key_points[i].response = rng.uniform(1.0, 100.0);
  }

  KeyPointsFilter::filterBest(key_points, 100);
  EXPECT_EQ(key_points.size(), 100);
  for (size_t i = 0; i < key_points.size() - 1; ++i)
  {
    EXPECT_GE(key_points[i].response, key_points[i + 1].response);
  }

  KeyPointsFilter::filterBest(key_points, 0);
  {
    EXPECT_EQ(key_points.size(), 0);
  }
}

