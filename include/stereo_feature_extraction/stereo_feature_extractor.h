#ifndef STEREO_FEATURE_EXTRACTOR_H
#define STEREO_FEATURE_EXTRACTOR_H

#include <opencv2/features2d/features2d.hpp>

#include <feature_extraction/feature_extractor.h>

#include "stereo_feature_set.h"
#include "stereo_camera_model.h"

namespace stereo_feature_extraction
{


/**
* \class StereoFeatureExtractor
* \brief Extractor for matching key points in a rectified stereo image pair
* Due to epipolar constraints the matching of key points from left to
* right image can be limited. Therefore StereoFeatureExtractor has
* some variables to control these constraints. See cpp file for default values.
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
    void setFeatureExtractor(const feature_extraction::FeatureExtractor::Ptr& feature_extractor);

    /**
    * \param model new camera model
    */
    void setCameraModel(const StereoCameraModel::Ptr& model);

    /**
    * \param match_method match method to use
    */
    void setMatchMethod(const MatchMethod& match_method);

    /**
    * \param max_y_diff the maximum difference in y-coordinates that
    *        matching key points can have. In the ideal case, this value
    *        is zero, due to camera calibration errors and noise it
    *        should be set to something around 0.5-2.0
    */
    void setMaxYDiff(double max_y_diff);

    /**
    * \param max_angle_diff the maximum allowed difference of the
    *        angles (directions) of two matching key points.
    */
    void setMaxAngleDiff(double max_angle_diff);

    /**
    * \param max_size_diff the maximum allowed difference of the
    *        size parameter of two matching key points.
    */
    void setMaxSizeDiff(int max_size_diff);

    /**
    * \param min_depth the minimum depth of world points to detect
    *        (search can be limited when this is high)
    */
    void setMinDepth(double min_depth);

    /**
    * \param max_depth the maximum depth of world points to detect
    *        (search can be limited when this is low)
    */
    void setMaxDepth(double max_depth);

    /**
    * \param roi region of interest where features have to be extracted,
    *        if width or height are zero, the whole image will be used (no roi)
    */
    void setRegionOfInterest(const cv::Rect& roi);

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
    * \return set of stereo features
    */
    StereoFeatureSet extract(const cv::Mat& image_left, 
            const cv::Mat& image_right) const;


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
            const std::vector<feature_extraction::KeyPoint>& key_points_left,
            const std::vector<feature_extraction::KeyPoint>& key_points_right,
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
    * Extracts stereo key points from given rectified stereo image pair.
    * Keypoints for each image are computed, a match mask that preserves
    * the epipolar constraints (given by internal parameters) is computed, 
    * descriptors for keypoints are computed and matched.
    * Afterwards, for each match a 3d point is computed based on a
    * stereo camera model.
    * \param image_left the left rectified image
    * \param image_right the right rectified image
     * \return set of stereo features
    */
    StereoFeatureSet extractKeyPointToKeyPoint(
            const cv::Mat& image_left, const cv::Mat& image_right) const;

    /**
    * Extracts stereo key points from given rectified stereo image pair
    * using a combination of key point detection and block matching.
    * Keypoints for the left image are computed and corresponding points
    * in the right image are found by block matching, preserving the given 
    * constraints.
    * Afterwards, for each match a 3d point is computed based on a
    * stereo camera model.
    * \param image_left the left rectified image
    * \param image_right the right rectified image
    * \param max_distance the block matching threshold
    * \return vector of stereo features
    */
    StereoFeatureSet extractKeyPointToBlock(const cv::Mat& image_left, 
            const cv::Mat& image_right, double max_distance) const;

    /**
    * Finds a correspondence using block matching
    * \param image_left the left image
    * \param point_left the left point which neighborhood is used as search 
    *        pattern
    * \param image_right the search image
    * \param distance the distance of the matching blocks is saved here
    */
    cv::Point2f findCorrespondenceBM(const cv::Mat& image_left,
        const cv::Point2f& point_left, const cv::Mat& image_right, 
        double* distance) const;

    feature_extraction::FeatureExtractor::Ptr feature_extractor_;
    StereoCameraModel::Ptr stereo_camera_model_;

    MatchMethod match_method_;

    double max_y_diff_;
    double max_angle_diff_;
    int max_size_diff_;
    double min_depth_;
    double max_depth_;

    cv::Rect region_of_interest_;

};

}

#endif


