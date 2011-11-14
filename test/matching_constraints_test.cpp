#include <gtest/gtest.h>
#include <opencv2/features2d/features2d.hpp>

#include <ros/package.h>

#include <vision_msgs/Features3D.h>

#include "feature_matching/matching_constraints.h"

TEST(MatchConstraints, disparityTest)
{
  std::vector<cv::KeyPoint> key_points_left(1);
  key_points_left[0].pt.x = 100.0;
  key_points_left[0].pt.y = 100.0;
  key_points_left[0].size = 0.0;
  key_points_left[0].angle = 0.0;
  key_points_left[0].response = 100.0;
  key_points_left[0].octave = 1;

  std::vector<cv::KeyPoint> key_points_right(1);
  key_points_right[0].pt.x = 110.0; // disparity = -10
  key_points_right[0].pt.y = 100.0;
  key_points_right[0].size = 0.0;
  key_points_right[0].angle = 0.0;
  key_points_right[0].response = 100.0;
  key_points_right[0].octave = 1;
  key_points_right.push_back(key_points_left[0]);

  double max_y_diff = 10.0;
  double max_angle_diff = 10.0;
  double max_size_diff = 10.0;
  double min_disparity = 0.0;
  double max_disparity = 100.0;
  cv::Mat match_mask;
  feature_matching::matching_constraints::computeStereoMatchMask(
      key_points_left, key_points_right, match_mask, max_y_diff, max_angle_diff,
      max_size_diff, min_disparity, max_disparity);

  EXPECT_EQ(match_mask.rows, key_points_left.size());
  EXPECT_EQ(match_mask.cols, key_points_right.size());

  EXPECT_EQ(match_mask.at<unsigned char>(0, 0), 0);
  EXPECT_EQ(match_mask.at<unsigned char>(0, 1), 255);
  
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

