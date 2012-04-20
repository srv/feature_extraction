#ifndef STEREO_FEATURE_MATCHER_H
#define STEREO_FEATURE_MATCHER_H

#include <opencv2/features2d/features2d.hpp>

namespace feature_matching
{
  /**
  * \brief Matcher for stereo key points w/ descriptors
  */
  class StereoFeatureMatcher
  {
    public:

      /**
      * \brief Parameters that control the epipolar constraints for matching
      */
      struct Params
      {
        /**
        * the maximum difference of the y coordinates of
        * left and right keypoints to be accepted as match candidate
        */
        double max_y_diff;

        /**
        * the maximum difference of the keypoint orientation
        * in degrees
        */
        double max_angle_diff;

        /**
        * the maximum difference of keypoint sizes to accept
        */
        double max_size_diff;

      };

      StereoFeatureMatcher() {}

      void setParams(const Params& params)
      {
        params_ = params;
      }

      /**
      * Computes the matches between left and right key_points/descriptors.
      * Epipolar constraints set in params are preserved.
      */
      void match(const std::vector<cv::KeyPoint>& key_points_left, 
                 const cv::Mat& descriptors_left,
                 const std::vector<cv::KeyPoint>& key_points_right,
                 const cv::Mat& descriptors_right,
                 double matching_threshold,
                 std::vector<cv::DMatch>& matches) const;

      /**
      * Computes a match candidate mask that fulfills epipolar constraints, i.e.
      * it is set to 255 for keypoint pairs that should be allowed to match and
      * 0 for all other pairs. 
      * \param key_points_left keypoints extracted from the left image
      * \param key_points_right keypoints extracted from the right image
      * \param match_mask matrix to store the result, will be allocated to
      *        rows = key_points_left.size(), cols = key_points_right.size()
      *        with type = CV_8UC1.
      */
      void computeMatchMask(const std::vector<cv::KeyPoint>& key_points_left,
                            const std::vector<cv::KeyPoint>& key_points_right,
                            cv::Mat& match_mask) const;

    private:

      Params params_;
  };

}

#endif

