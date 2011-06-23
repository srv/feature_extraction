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

    enum MatchMethod
    {
        KEY_POINT_TO_KEY_POINT,
        KEY_POINT_TO_BLOCK
    };

    /**
    * Constructs an empty stereo feature extractor. Make sure to call
    * setFeatureExtractor and setCameraModel before usage.
    */
    StereoFeatureExtractor();

    /**
    * \param feature_extractor new feature extractor to use
    */
    void setFeatureExtractor(const FeatureExtractor::Ptr& feature_extractor);

    /**
    * \param match_method match method to use
    */
    void setMatchMethod(const MatchMethod& match_method);

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
    *
    * \param image_left the left rectified image
    * \param image_right the right rectified image
    * \param max_y_diff the maximum difference of the y coordinates of
    *        left and right keypoints to be accepted as match candidate
    * \param max_angle_diff the maximum difference of the keypoint orientation
    *        in degrees
    * \param max_size_diff the maximum difference of keypoint sizes to accept
    * \param min_depth the minimum depth (possible distance to an object)
    *        limits the disparity and therefore the correspondence search range
    * \param max_depth the maximum depth (possible distance to an object)
    * \return vector of stereo features
    */
    std::vector<StereoFeature> extract(const cv::Mat& image_left, 
            const cv::Mat& image_right, double max_y_diff, 
            double max_angle_diff, int max_size_diff,
            double min_depth, double max_depth) const;


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
            const std::vector<KeyPoint>& key_points_left,
            const std::vector<KeyPoint>& key_points_right,
            cv::Mat& match_mask, double max_y_diff,
            double max_angle_diff, int max_size_diff,
            double min_disparity, double max_disparity);

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

    /**
    * Extracts stereo keypoints from given rectified stereo image pair.
    * Keypoints for each image are computed, a match mask that preserves
    * the epipolar constraints (given by max* parameters) is computed, 
    * descriptors for keypoints are computed and matched.
    * Afterwards, for each match a 3d point is computed based on a
    * stereo camera model.
    * \param image_left the left rectified image
    * \param image_right the right rectified image
    * \param max_y_diff the maximum difference of the y coordinates of
    *        left and right keypoints to be accepted as match candidate
    * \param max_angle_diff the maximum difference of the keypoint orientation
    *        in degrees
    * \param max_size_diff the maximum difference of keypoint sizes to accept
    * \param min_depth the minimum depth (possible distance to an object)
    *        limits the disparity and therefore the correspondence search range
    * \param max_depth the maximum depth (possible distance to an object)
     * \return vector of stereo features
    */
    std::vector<StereoFeature> extractKeyPointToKeyPoint(const cv::Mat& image_left, 
            const cv::Mat& image_right, double max_y_diff, 
            double max_angle_diff, int max_size_diff,
            double min_depth, double max_depth) const;

    /**
    * Extracts stereo keypoints from given rectified stereo image pair
    * using a combination of key point detection and block matching.
    * Keypoints for the left image are computed and corresponding points
    * in the right image are found by block matching, preserving the given 
    * constraints.
    * Afterwards, for each match a 3d point is computed based on a
    * stereo camera model.
    * \param image_left the left rectified image
    * \param image_right the right rectified image
    * \param max_y_diff the maximum difference of the y coordinates of
    *        left and right points to be accepted as match
    * \param max_distance the block matching threshold
    * \return vector of stereo features
    */
    std::vector<StereoFeature> extractKeyPointToBlock(const cv::Mat& image_left, 
            const cv::Mat& image_right,  double max_y_diff, 
            double max_distance) const;

    /**
    * Finds a correspondence using block matching
    * \param image_left the left image
    * \param point_left the left point which neighborhood is used as search 
    *        pattern
    * \param image_right the search image
    * \param max_y_dist tha maximum distance in y that may have the matching points
    * \param distance the distance of the matching blocks is saved here
    */
    static cv::Point2f findCorrespondenceBM(const cv::Mat& image_left,
        const cv::Point2f& point_left, const cv::Mat& image_right, 
        double max_y_dist, double* distance);

    FeatureExtractor::Ptr feature_extractor_;
    StereoCameraModel::Ptr stereo_camera_model_;

    MatchMethod match_method_;

};

}

#endif


