#ifndef MATCHING_METHODS_H_
#define MATCHING_METHODS_H_

#include <opencv2/features2d/features2d.hpp>

namespace feature_matching
{
namespace matching_methods
{
/**
 * Matches two sets of descriptors, searching for the 2 nearest neighbors
 * in train_descriptors for each element in query_descriptors. Matches are accepted
 * if the ratio of the distance from the first match and the distance from
 * the second match are below the given threshold or if only one neighbor
 * was found.
 * \param query_descriptors query descriptors
 * \param train_descriptors reference descriptors
 * \param threshold the matching threshold
 * \param match_mask the mask to use to allow matches, if empty, all
 *        descriptors are matched to each other, if not empty,
 *        must be of size descriptors1.rows * descriptors2.rows
 * \param matches vector to store matches
 */
void thresholdMatching(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors,
                       double threshold, const cv::Mat& match_mask,
                       std::vector<cv::DMatch>& matches);

/**
* Filters matches so that checked_matches contains only those matches that
* occure both in matches1to2 and matches2to1.
*/
void crossCheckFilter(const std::vector<cv::DMatch>& matches1to2,
                      const std::vector<cv::DMatch>& matches2to1,
                      std::vector<cv::DMatch>& checked_matches);
/**
 * Combines the above two methods, matching query_descriptors to train_descriptors
 * and train_descriptors to query_descriptors using thresholdMatching and filtering
 * the result using crossCheckFilter.
 * \param query_descriptors query descriptors
 * \param train_descriptors reference descriptors
 * \param threshold the matching threshold
 * \param match_mask the mask to use to allow matches, if empty, all
 *        descriptors are matched to each other, if not empty,
 *        must be of size descriptors1.rows * descriptors2.rows
 * \param matches vector to store matches
 */
void crossCheckThresholdMatching(const cv::Mat& query_descriptors, 
                       const cv::Mat& train_descriptors,
                       double threshold, const cv::Mat& match_mask,
                       std::vector<cv::DMatch>& matches);
}
}

#endif


