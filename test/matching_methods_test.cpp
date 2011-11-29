#include <gtest/gtest.h>

#include <opencv2/highgui/highgui.hpp>

#include <ros/package.h>

#include "feature_matching/matching_methods.h"

TEST(MatchingMethods, crossCheckFilterTest)
{
  std::vector<cv::DMatch> forward_matches(100);
  std::vector<cv::DMatch> backward_matches(forward_matches.size());
  for (size_t i = 0; i < forward_matches.size(); ++i)
  {
    forward_matches[i].trainIdx = i;
    forward_matches[i].queryIdx = i+1;
    backward_matches[i].trainIdx = forward_matches[i].queryIdx;
    backward_matches[i].queryIdx = forward_matches[i].trainIdx;
  }

  std::vector<cv::DMatch> filtered_matches;
  feature_matching::matching_methods::crossCheckFilter(forward_matches, 
                                                    backward_matches, 
                                                    filtered_matches);

  ASSERT_EQ(forward_matches.size(), filtered_matches.size());
  for (size_t i = 0; i < forward_matches.size(); ++i)
  {
    EXPECT_EQ(forward_matches[i].trainIdx, filtered_matches[i].trainIdx);
    EXPECT_EQ(forward_matches[i].queryIdx, filtered_matches[i].queryIdx);
  }

  feature_matching::matching_methods::crossCheckFilter(forward_matches,
                                                    forward_matches,
                                                    filtered_matches);

  EXPECT_EQ(filtered_matches.size(), 0);

}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

