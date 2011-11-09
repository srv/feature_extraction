#ifndef STEREO_MATCHING_H_
#define STEREO_MATCHING_H_

#include <opencv2/features2d/features2d.hpp>

namespace stereo_feature_extraction
{
  namespace stereo_matching
  {
    /**
    * Computes a match candidate mask that fulfills epipolar constraints, i.e.
    * it is set to 255 for keypoint pairs that should be allowed to match and
    * 0 for all other pairs. Keypoints with different octaves will not be
    * allowed to match.
    * \param key_points_left keypoints extracted from the left image
    * \param key_points_right keypoints extracted from the right image
    * \param match_mask matrix to store the result, will be allocated to
    *        rows = key_points_left.size(), cols = key_points_right.size()
    *        with type = CV_8UC1.
    * \param max_y_diff the maximum difference of the y coordinates of
    *        left and right keypoints to be accepted as match candidate
    * \param max_angle_diff the maximum difference of the keypoint orientation
    *        in degrees
    * \param max_size_diff the maximum difference of keypoint sizes to accept
    * \param min_disparity minimum allowed disparity
    * \param max_disparity maximum allowed disparity
    *
    */
    void computeMatchMask(
            const std::vector<cv::KeyPoint>& key_points_left,
            const std::vector<cv::KeyPoint>& key_points_right,
            cv::Mat& match_mask, double max_y_diff,
            double max_angle_diff, int max_size_diff,
            double min_disparity, double max_disparity);


    /**
    * Matches two sets of descriptors, searching for the 2 nearest neighbors
    * in descriptors2 for each element in descriptors1. Matches are accepted
    * if the ratio of the distance from the first match and the distance from
    * the second match are below the given threshold.
    * \param descriptors1 reference descriptors
    * \param descriptors2 test descriptors
    * \param matches vector to store matches
    * \param threshold the matching threshold
    * \param match_mask the mask to use to allow matches, if empty, all
    *        descriptors are matched to each other
    */
    void thresholdMatching(const cv::Mat& descriptors1, const cv::Mat& descriptors2, 
        std::vector<cv::DMatch>& matches, double threshold, const cv::Mat& match_mask = cv::Mat());

  }
}

#endif


