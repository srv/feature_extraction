
#include <iostream>
#include "feature_matching/matching_methods.h"

void feature_matching::matching_methods::thresholdMatching(
    const cv::Mat& query_descriptors, const cv::Mat& train_descriptors,
    double threshold, const cv::Mat& match_mask,
    std::vector<cv::DMatch>& matches)
{
  matches.clear();
  if (query_descriptors.empty() || train_descriptors.empty())
    return;
  assert(query_descriptors.type() == train_descriptors.type());
  assert(query_descriptors.cols == train_descriptors.cols);

  const int knn = 2;
  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher;
  // choose matcher based on feature type
  if (query_descriptors.type() == CV_8U)
  {
    descriptor_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  }
  else
  {
    descriptor_matcher = cv::DescriptorMatcher::create("BruteForce");
  }
  std::vector<std::vector<cv::DMatch> > knn_matches;
  descriptor_matcher->knnMatch(query_descriptors, train_descriptors,
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

void feature_matching::matching_methods::crossCheckThresholdMatching(
  const cv::Mat& query_descriptors, const cv::Mat& train_descriptors,
  double threshold, const cv::Mat& match_mask,
  std::vector<cv::DMatch>& matches)
{
  std::vector<cv::DMatch> query_to_train_matches;
  thresholdMatching(query_descriptors, train_descriptors, threshold, match_mask, query_to_train_matches);
  std::vector<cv::DMatch> train_to_query_matches;
  cv::Mat match_mask_t;
  if (!match_mask.empty()) match_mask_t = match_mask.t();
  thresholdMatching(train_descriptors, query_descriptors, threshold, match_mask_t, train_to_query_matches);

  crossCheckFilter(query_to_train_matches, train_to_query_matches, matches);
}

