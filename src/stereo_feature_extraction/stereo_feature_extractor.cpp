#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include "stereo_feature_extractor.h"
#include "feature_extractor_factory.h"

using namespace stereo_feature_extraction;

StereoFeatureExtractor::StereoFeatureExtractor()
{
}

StereoFeatureExtractor::StereoFeatureExtractor(
        const FeatureExtractor::Ptr& feature_extractor,
        const StereoCameraModel::Ptr& camera_model) :
    feature_extractor_(feature_extractor),
    stereo_camera_model_(camera_model)
{
}

void StereoFeatureExtractor::setFeatureExtractor(
        const FeatureExtractor::Ptr& feature_extractor)
{
    feature_extractor_ = feature_extractor;
}

void StereoFeatureExtractor::setCameraModel(
        const StereoCameraModel::Ptr& model)
{
    stereo_camera_model_ = model;
}

std::vector<StereoFeature> StereoFeatureExtractor::extract(
        const cv::Mat& image_left, const cv::Mat& image_right, 
        const cv::Mat& mask_left, const cv::Mat& mask_right,
        double max_y_diff, double max_angle_diff, 
        int max_size_diff) const
{
    assert(feature_extractor_.get() != NULL);
    assert(stereo_camera_model_.get() != NULL);

    std::vector<KeyPoint> key_points_left;
    cv::Mat descriptors_left;
    feature_extractor_->extract(image_left, mask_left, key_points_left, 
            descriptors_left);

    std::vector<KeyPoint> key_points_right;
    cv::Mat descriptors_right;
    feature_extractor_->extract(image_right, mask_right, key_points_right,
            descriptors_right);

    cv::Mat match_mask;
    computeMatchMask(key_points_left, key_points_right, match_mask, max_y_diff,
            max_angle_diff, max_size_diff);

    std::vector<cv::DMatch> matches;
    crossCheckMatching(descriptors_left, descriptors_right, matches, match_mask);

    std::vector<StereoFeature> stereo_features(matches.size());
    for (size_t i = 0; i < matches.size(); ++i)
    {
        int index_left = matches[i].queryIdx;
        int index_right = matches[i].trainIdx;
        const KeyPoint& key_point_left = key_points_left[index_left];
        const KeyPoint& key_point_right = key_points_right[index_right];
        stereo_features[i].world_point = 
            stereo_camera_model_->computeWorldPoint(key_point_left.pt, 
                                                    key_point_right.pt);
        stereo_features[i].key_point = key_point_left;
        cv::Mat descriptor = descriptors_left.row(index_left);
        descriptor.copyTo(stereo_features[i].descriptor);
        stereo_features[i].color = 
            image_left.at<cv::Vec3b>(key_point_left.pt.y,
                                     key_point_left.pt.x);
    }
    return stereo_features;
}


void StereoFeatureExtractor::computeMatchMask(
        const std::vector<KeyPoint>& key_points_left,
        const std::vector<KeyPoint>& key_points_right,
        cv::Mat& match_mask, double max_y_diff, double max_angle_diff, 
        int max_size_diff)
{
    if (key_points_left.empty() || key_points_right.empty())
    {
        return;
    }

    match_mask.create(key_points_right.size(), key_points_left.size(), CV_8UC1);
    for (int r = 0; r < match_mask.rows; ++r)
    {
        for (int c = 0; c < match_mask.cols; ++c)
        {
            const KeyPoint& keypoint1 = key_points_left[c];
            const KeyPoint& keypoint2 = key_points_right[r];
            double y_diff = fabs(keypoint1.pt.y - keypoint2.pt.y);
            double disparity = keypoint1.pt.x - keypoint2.pt.x;
            double angle_diff = std::abs(keypoint1.angle - keypoint2.angle);
            angle_diff = std::min(360 - angle_diff, angle_diff);
            int size_diff = std::abs(keypoint1.size - keypoint2.size);
            if (y_diff <= max_y_diff && disparity > 0 && angle_diff <= max_angle_diff && 
                keypoint1.octave == keypoint2.octave && size_diff <= max_size_diff)
            {
                match_mask.at<unsigned char>(r, c) = 255;
            }
            else
            {
                match_mask.at<unsigned char>(r, c) = 0;
            }
        }
    }
}

// TODO remove distance threshold hack!!
void StereoFeatureExtractor::crossCheckMatching(
                            const cv::Mat& descriptors_left, 
                            const cv::Mat& descriptors_right,
                            std::vector<cv::DMatch>& matches, 
                            const cv::Mat& match_mask)
{
    matches.clear();
    int knn = 1;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = 
        cv::DescriptorMatcher::create("BruteForce");
    std::vector<std::vector<cv::DMatch> > matches_left2right, matches_right2left;
    descriptor_matcher->knnMatch(descriptors_left, descriptors_right,
            matches_left2right, knn, match_mask.t());
    descriptor_matcher->knnMatch(descriptors_right, descriptors_left,
            matches_right2left, knn, match_mask);
    static double DISTANCE_THRESHOLD = 0.5;
    for (size_t m = 0; m < matches_left2right.size(); m++ )
    {
        bool cross_check_found = false;
        for (size_t fk = 0; fk < matches_left2right[m].size(); fk++ )
        {
            const cv::DMatch& forward = matches_left2right[m][fk];

            for( size_t bk = 0; bk < matches_right2left[forward.trainIdx].size(); bk++ )
            {
                const cv::DMatch& backward = matches_right2left[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    if (forward.distance < DISTANCE_THRESHOLD)
                    {
                        matches.push_back(forward);
                        cross_check_found = true;
                    }
                    break;
                }
            }
            if (cross_check_found) break;
        }
    }
}

