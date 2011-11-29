#ifndef MATCHING_METHODS_H_
#define MATCHING_METHODS_H_

#include <opencv2/features2d/features2d.hpp>

namespace feature_matching
{
  namespace matching_methods
  {
    /**
    * Matches two sets of descriptors, searching for the 2 nearest neighbors
    * in descriptors2 for each element in descriptors1. Matches are accepted
    * if the ratio of the distance from the first match and the distance from
    * the second match are below the given threshold or if only one neighbor was found.
    * \param descriptors1 reference descriptors
    * \param descriptors2 test descriptors
    * \param matches vector to store matches
    * \param threshold the matching threshold
    * \param match_mask the mask to use to allow matches, if empty, all
    *        descriptors are matched to each other, if not empty, 
    *        must be of size descriptors1.rows * descriptors2.rows
    */
    void thresholdMatching(const cv::Mat& descriptors1, const cv::Mat& descriptors2, 
                           double threshold, const cv::Mat& match_mask, std::vector<cv::DMatch>& matches);

    void crossCheckFilter(const std::vector<cv::DMatch>& matches1to2, 
                          const std::vector<cv::DMatch>& matches2to1,
                          std::vector<cv::DMatch>& checked_matches);
  }
}

#endif


