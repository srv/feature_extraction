#ifndef STEREO_FEATURE_MATCHER_H_
#define STEREO_FEATURE_MATCHER_H_

#include <opencv2/features2d/features2d.hpp>

#include "stereo_camera_model.h"

namespace stereo_feature_extraction
{

/**
* \class StereoFeatureMatcher
* \brief Matcher for matching key points in a rectified stereo image pair
* Due to epipolar constraints the matching of key points from left to
* right image can be limited. Therefore this class has
* some parameters to control these constraints. See the member documentation 
* below for a description of these variables, see cpp file for default values.
*/
class StereoFeatureMatcher
{

  public:

    StereoFeatureMatcher();

    /**
    * Updates the camera information (stereo camera model), be sure to call this
    * method at least once before the use of match().
    */
    inline void setStereoCameraModel(const StereoCameraModel& model) { stereo_camera_model_ = model; }

    inline void setMaxYDiff(double max_y_diff) { max_y_diff_ = max_y_diff; }

    inline void setMaxAngleDiff(double max_angle_diff) { max_angle_diff_ = max_angle_diff; }

    inline void setMaxSizeDiff(int max_size_diff) { max_size_diff_ = max_size_diff; }
    
    inline void setMinDepth(double min_depth) { min_depth_ = min_depth; }

    inline void setMaxDepth(double max_depth) { max_depth_ = max_depth; }

    /**
    * The main matching method.
    * Matches left and right features that fulfill epipolar constraints.
    */
    Features3D match(const Features& features_left, const Features& features_right) const;

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
    static void computeMatchMask(
            const std::vector<cv::KeyPoint>& key_points_left,
            const std::vector<cv::KeyPoint>& key_points_right,
            cv::Mat& match_mask, double max_y_diff,
            double max_angle_diff, int max_size_diff,
            double min_disparity, double max_disparity);

    /**
    * Matches two sets of descriptors, searching for the 2 nearest neighbors
    * in features_right for each element in features_left. Matches are accepted
    * if the ratio of the distance from the first match and the distance from
    * the second match are below a threshold.
    * \param features_left descriptors for left image
    * \param features_right descriptors for right image
    * \param matches vector to store matches
    * \param match_mask the mask to use to allow matches, if empty, all
    *        descriptors are matched to each other
    */
    static void thresholdMatching(
            const cv::Mat& descriptors_left, 
            const cv::Mat& descriptors_right,
            std::vector<cv::DMatch>& matches, 
            const cv::Mat& match_mask = cv::Mat());

  protected:

    /// the model for the stereo camera system
    StereoCameraModel stereo_camera_model_;

    /// the maximum difference in y-coordinates that
    /// matching key points can have. In the ideal case, this value
    /// is zero, due to camera calibration errors and noise it
    /// should be set to something around 0.5-2.0
    double max_y_diff_;

    /// max_angle_diff the maximum allowed difference of the
    /// angles (directions) of two matching key points.
    double max_angle_diff_;

    /// max_size_diff the maximum allowed difference of the
    /// size parameter of two matching key points.
    int max_size_diff_;

    /// min_depth the minimum depth (z-value) in meters of world points
    /// (search can be limited when this is high)
    double min_depth_;

    /// max_depth the maximum depth (z-value) in meters of world points
    /// (search can be limited when this is low)
    double max_depth_;
};

}

#endif


