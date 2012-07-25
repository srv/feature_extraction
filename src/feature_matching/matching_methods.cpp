
#include <iostream>
#include "feature_matching/matching_methods.h"

void feature_matching::matching_methods::thresholdMatching(
    const cv::Mat& descriptors1, const cv::Mat& descriptors2,
    double threshold, const cv::Mat& match_mask,
    std::vector<cv::DMatch>& matches)
{
  assert(descriptors1.type() == descriptors2.type());
  assert(descriptors1.cols == descriptors2.cols);

  matches.clear();
  int knn = 2;
  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher;
  if (descriptors1.type() == CV_8U)
  {
    descriptor_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  }
  else
  {
    descriptor_matcher = cv::DescriptorMatcher::create("BruteForce");
  }
  std::vector<std::vector<cv::DMatch> > knn_matches;
  descriptor_matcher->knnMatch(descriptors1, descriptors2,
          knn_matches, knn);

  for (size_t m = 0; m < knn_matches.size(); m++ )
  {
    if (knn_matches[m].size() < 2) continue;
    bool match_allowed = match_mask.empty() ? true : match_mask.at<unsigned char>(
        knn_matches[m][0].queryIdx, knn_matches[m][0].trainIdx) > 0;
    float dist1 = knn_matches[m][0].distance;
    float dist2 = knn_matches[m][1].distance;
    if (dist1 / dist2 < threshold && match_allowed)
    {
      matches.push_back(knn_matches[m][0]);
    }
  }
}

void feature_matching::matching_methods::crossCheckFilter(
    const std::vector<cv::DMatch>& matches1to2, 
    const std::vector<cv::DMatch>& matches2to1,
    std::vector<cv::DMatch>& checked_matches)
{
  checked_matches.clear();
  for (size_t i = 0; i < matches1to2.size(); ++i)
  {
    bool match_found = false;
    const cv::DMatch& forward_match = matches1to2[i];
    for (size_t j = 0; j < matches2to1.size() && match_found == false; ++j)
    {
      const cv::DMatch& backward_match = matches2to1[j];
      if (forward_match.trainIdx == backward_match.queryIdx &&
          forward_match.queryIdx == backward_match.trainIdx)
      {
        checked_matches.push_back(forward_match);
        match_found = true;
      }
    }
  }
}

