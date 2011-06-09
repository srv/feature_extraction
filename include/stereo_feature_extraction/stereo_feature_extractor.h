#ifndef STEREO_FEATURE_EXTRACTOR_H
#define STEREO_FEATURE_EXTRACTOR_H

#include <opencv2/features2d/features2d.hpp>

#include "stereo_feature.h"
#include "feature_extractor.h"
#include "stereo_camera_model.h"

namespace stereo_feature_extraction
{


/**
* \class StereoFeatureExtractor
* \brief Extractor for matching keypoints in a rectified stereo image pair
*/
class StereoFeatureExtractor
{

  public:

    /**
    * Constructs an empty stereo feature extractor. Make sure to call
    * setFeatureExtractor and setCameraModel before usage.
    */
    StereoFeatureExtractor();

    /**
    * Constructs a stereo feature extractor with given parameters:
    * \param feature_extractor the extractor to use to extract features from
    *        each stereo image
    * \param model the camera model that is used to compute the 3d
    *        position of features
    */
    StereoFeatureExtractor(const FeatureExtractor::Ptr& feature_extractor,
            const StereoCameraModel::Ptr& model);

    /**
    * \param feature_extractor new feature extractor to use
    */
    void setFeatureExtractor(const FeatureExtractor::Ptr& feature_extractor);

    /**
    * \param model new camera model
    */
    void setCameraModel(const StereoCameraModel::Ptr& model);

    /**
    * Extracts stereo keypoints from given rectified stereo image pair.
    * Keypoints for each image are computed, a match mask that preserves
    * the epipolar constraints (given by max* parameters) is computed, 
    * descriptors for keypoints are computed and matched.
    * Afterwards, for each match a 3d point is computed based on a
    * stereo camera model.
    * \param image_left the left rectified image
    * \param image_right the right rectified image
    * \param mask_left the mask where to extract key points seen in left image
    * \param mask_right the mask where to extract key points seen in right image
    * \param max_y_diff the maximum difference of the y coordinates of
    *        left and right keypoints to be accepted as match candidate
    * \param max_angle_diff the maximum difference of the keypoint orientation
    *        in degrees
    * \param max_size_diff the maximum difference of keypoint sizes to accept
    * \return vector of stereo features
    */
    std::vector<StereoFeature> extract(const cv::Mat& image_left, 
            const cv::Mat& image_right, const cv::Mat& mask_left, 
            const cv::Mat& mask_right, double max_y_diff, 
            double max_angle_diff, int max_size_diff) const;

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
    *
    */
    static void computeMatchMask(
            const std::vector<KeyPoint>& key_points_left,
            const std::vector<KeyPoint>& key_points_right,
            cv::Mat& match_mask, double max_y_diff,
            double max_angle_diff, int max_size_diff);

    /**
    * Matches two sets of descriptors using cross check, i.e. a match
    * is added for each pair that was matched from left to right AND from
    * right to left.
    * \param features_left descriptors for left image
    * \param features_right descriptors for right image
    * \param matches vector to store matches
    * \param match_mask the mask to use to allow matches, if empty, all
    *        descriptors are matched to each other
    */
    static void crossCheckMatching(
            const cv::Mat& descriptors_left, 
            const cv::Mat& descriptors_right,
            std::vector<cv::DMatch>& matches, 
            const cv::Mat& match_mask = cv::Mat());

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

    FeatureExtractor::Ptr feature_extractor_;
    StereoCameraModel::Ptr stereo_camera_model_;

};

}

#endif


