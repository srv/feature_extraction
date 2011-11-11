
#include <iostream>
#include "feature_matching/match_methods.h"

void feature_matching::match_methods::thresholdMatching(
                            const cv::Mat& descriptors1, 
                            const cv::Mat& descriptors2,
                            std::vector<cv::DMatch>& matches, 
                            double threshold,
                            const cv::Mat& match_mask)
{
  matches.clear();
  int knn = 2;
  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = 
      cv::DescriptorMatcher::create("BruteForce");
  std::vector<std::vector<cv::DMatch> > matches_1to2;
  descriptor_matcher->knnMatch(descriptors1, descriptors2,
          matches_1to2, knn);

  std::cout << matches_1to2.size() << " matches before filtering." << std::endl;
  for (size_t m = 0; m < matches_1to2.size(); m++ )
  {
    if (matches_1to2[m].size() == 2) 
    {
      float dist1 = matches_1to2[m][0].distance;
      float dist2 = matches_1to2[m][1].distance;
      int queryIndex = matches_1to2[m][0].queryIdx;
      int trainIndex = matches_1to2[m][0].trainIdx;
      if (dist1 / dist2 < threshold && match_mask.at<unsigned char>(trainIndex, queryIndex) > 0)
      {
        matches.push_back(matches_1to2[m][0]);
      }
    }
    else if (matches_1to2[m].size() == 1) // only one match found -> save it
    {
      matches.push_back(matches_1to2[m][0]);
    }
  }
  std::cout << matches.size() << " matches after filtering." << std::endl;
}

