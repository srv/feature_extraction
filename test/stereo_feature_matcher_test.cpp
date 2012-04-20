#include <gtest/gtest.h>

#include <opencv2/highgui/highgui.hpp>

#include <ros/package.h>

#include "feature_matching/stereo_feature_matcher.h"

TEST(StereoFeatureMatcher, disparityTest)
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

  feature_matching::StereoFeatureMatcher::Params params;
  params.max_y_diff = 10.0;
  params.max_angle_diff = 10.0;
  params.max_size_diff = 10.0;
  feature_matching::StereoFeatureMatcher matcher;
  matcher.setParams(params);

  cv::Mat match_mask;
  matcher.computeMatchMask(
      key_points_left, key_points_right, match_mask);

  EXPECT_EQ(match_mask.rows, key_points_left.size());
  EXPECT_EQ(match_mask.cols, key_points_right.size());

  EXPECT_EQ(match_mask.at<unsigned char>(0, 0), 0);
  EXPECT_EQ(match_mask.at<unsigned char>(0, 1), 255);
  
}

TEST(StereoFeatureMatcher, emptyTest)
{
  std::vector<cv::KeyPoint> key_points_left;
  std::vector<cv::KeyPoint> key_points_right;

  feature_matching::StereoFeatureMatcher::Params params;
  params.max_y_diff = 10.0;
  params.max_angle_diff = 10.0;
  params.max_size_diff = 10.0;
  feature_matching::StereoFeatureMatcher matcher;
  matcher.setParams(params);

  cv::Mat match_mask;
  matcher.computeMatchMask(
      key_points_left, key_points_right, match_mask);

  EXPECT_EQ(match_mask.rows, 0);
  EXPECT_EQ(match_mask.cols, 0);
}

TEST(StereoFeatureMatcher, matchTestEmpty)
{
  std::vector<cv::KeyPoint> key_points_left;
  cv::Mat descriptors_left;
  std::vector<cv::KeyPoint> key_points_right;
  cv::Mat descriptors_right;

  feature_matching::StereoFeatureMatcher::Params params;
  params.max_y_diff = 10.0;
  params.max_angle_diff = 10.0;
  params.max_size_diff = 10.0;
  feature_matching::StereoFeatureMatcher matcher;
  matcher.setParams(params);

  std::vector<cv::DMatch> matches;

  matcher.match(
      key_points_left, descriptors_left,
      key_points_right, descriptors_right,
      0.8, matches);

  EXPECT_EQ(matches.size(), 0);
}




// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

