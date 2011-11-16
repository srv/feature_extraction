
#include <iostream>
#include "feature_matching/matching_methods.h"

void feature_matching::matching_methods::thresholdMatching(
                            const cv::Mat& descriptors1, 
                            const cv::Mat& descriptors2,
                            double threshold,
                            const cv::Mat& match_mask,
                            std::vector<cv::DMatch>& matches)
{
  matches.clear();
  int knn = 2;
  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = 
      cv::DescriptorMatcher::create("BruteForce");
  std::vector<std::vector<cv::DMatch> > knn_matches;
  descriptor_matcher->knnMatch(descriptors1, descriptors2,
          knn_matches, knn, match_mask);

  // output for debugging
  // std::cout << knn_matches.size() << " matches before threshold filtering." << std::endl;
  int two_found, one_found, zero_found;
  two_found = one_found = zero_found = 0;
  for (size_t m = 0; m < knn_matches.size(); m++ )
  {
    if (knn_matches[m].size() == 2) 
    {
      float dist1 = knn_matches[m][0].distance;
      float dist2 = knn_matches[m][1].distance;
      if (dist1 / dist2 < threshold)
      {
        matches.push_back(knn_matches[m][0]);
      }
      two_found++;
      /*
      // output for debugging
      if (m % (knn_matches.size() / 5) == 0)
      {
        std::cout << dist1 << " / " << dist2 << " = " << dist1 / dist2 << std::endl;
      }
      */
    }
    else if (knn_matches[m].size() == 1) // only one match found -> save it
    {
      matches.push_back(knn_matches[m][0]);
      one_found++;
    }
    else
    {
      zero_found++;
    }
  }
  // output for debugging
  // std::cout << matches.size() << " matches after threshold filtering, " << two_found << " with k=2, " << one_found << " with k=1 " << zero_found << " had no partner." << std::endl;
}

