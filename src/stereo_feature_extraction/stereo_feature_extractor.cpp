#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include "stereo_feature_extractor.h"
#include "feature_extractor_factory.h"

using namespace stereo_feature_extraction;


static const double DEFAULT_MAX_Y_DIFF = 0.5;
static const double DEFAULT_MAX_ANGLE_DIFF = 5.0;
static const int DEFAULT_MAX_SIZE_DIFF = 0;
static const double DEFAULT_MIN_DEPTH = 0.1;
static const double DEFAULT_MAX_DEPTH = 10.0;
static const StereoFeatureExtractor::MatchMethod DEFAULT_MATCH_METHOD = 
  StereoFeatureExtractor::KEY_POINT_TO_KEY_POINT;

StereoFeatureExtractor::StereoFeatureExtractor() :
    match_method_(DEFAULT_MATCH_METHOD),
    max_y_diff_(DEFAULT_MAX_Y_DIFF),
    max_angle_diff_(DEFAULT_MAX_ANGLE_DIFF),
    max_size_diff_(DEFAULT_MAX_SIZE_DIFF),
    min_depth_(DEFAULT_MIN_DEPTH),
    max_depth_(DEFAULT_MAX_DEPTH)
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

void StereoFeatureExtractor::setMatchMethod(const MatchMethod& match_method)
{
    match_method_ = match_method;
}

void StereoFeatureExtractor::setMaxYDiff(double max_y_diff)
{
    max_y_diff_ = max_y_diff;
}

void StereoFeatureExtractor::setMaxAngleDiff(double max_angle_diff)
{
    max_angle_diff_ = max_angle_diff;
}

void StereoFeatureExtractor::setMaxSizeDiff(int max_size_diff)
{
    max_size_diff_ = max_size_diff;
}

void StereoFeatureExtractor::setMinDepth(double min_depth)
{
    min_depth_ = min_depth;
}

void StereoFeatureExtractor::setMaxDepth(double max_depth)
{
    max_depth_ = max_depth;
}

void StereoFeatureExtractor::setRegionOfInterest(const cv::Rect& roi)
{
    assert(roi.x >= 0 && roi.y >= 0 && roi.width >= 0 && roi.height >= 0);
    region_of_interest_ = roi;
}


std::vector<StereoFeature> StereoFeatureExtractor::extractKeyPointToBlock(
        const cv::Mat& image_left, const cv::Mat& image_right, 
        double max_distance) const
{
    assert(feature_extractor_.get() != NULL);
    assert(stereo_camera_model_.get() != NULL);

    std::vector<KeyPoint> key_points_left;
    cv::Mat descriptors_left;
    cv::Rect roi_left = region_of_interest_;
    if (roi_left.x + roi_left.width > image_left.cols) 
        roi_left.width = image_left.cols - roi_left.x;
    if (roi_left.y + roi_left.height > image_left.rows)
        roi_left.height = image_left.rows - roi_left.y;
    feature_extractor_->extract(image_left, key_points_left,
            descriptors_left, roi_left);

    std::vector<StereoFeature> stereo_features;
    for (size_t i = 0; i < key_points_left.size(); ++i)
    {
        const KeyPoint& key_point_left = key_points_left[i];
        double distance;
        cv::Point2f point_right = findCorrespondenceBM(image_left, 
                key_point_left.pt, image_right, &distance);
        if (distance < max_distance)
        {
            StereoFeature stereo_feature;
            stereo_feature.world_point = 
                stereo_camera_model_->computeWorldPoint(key_point_left.pt, 
                                                        point_right);
            stereo_feature.key_point_left = key_point_left;
            cv::Mat descriptor = descriptors_left.row(i);
            descriptor.copyTo(stereo_feature.descriptor);
            stereo_feature.color = 
                image_left.at<cv::Vec3b>(key_point_left.pt.y,
                                        key_point_left.pt.x);
            stereo_features.push_back(stereo_feature);
        }
    }
    return stereo_features;
}

cv::Point2f StereoFeatureExtractor::findCorrespondenceBM(const cv::Mat& image_left,
        const cv::Point2f& point_left, const cv::Mat& image_right, 
        double* distance) const
{
    int template_size = 15;
    int half_template_size = template_size / 2 + 1;
    if (point_left.x > half_template_size &&
        point_left.y > half_template_size &&
        point_left.x < image_left.cols - half_template_size &&
        point_left.y < image_left.rows - half_template_size)
    {
        int template_x = static_cast<int>(point_left.x - half_template_size);
        int template_y = static_cast<int>(point_left.y - half_template_size);
        cv::Rect template_roi(template_x, template_y, template_size, template_size);
        cv::Mat match_template = cv::Mat(image_left, template_roi);
        
        cv::Rect search_roi(0, point_left.y - max_y_diff_ - half_template_size, image_right.cols,
                2.0 * max_y_diff_ + template_size);
        cv::Mat search_image(image_right, search_roi);

        cv::Mat result_image;
        cv::matchTemplate(search_image, match_template, result_image, CV_TM_SQDIFF);

        double min_val, max_val;
        cv::Point max_loc;
        cv::Point min_loc;
        cv::minMaxLoc(result_image, &min_val, &max_val, &min_loc, &max_loc);

        /*
        cv::Point paint_point = min_loc;

        cv::Mat canvas = search_image.clone();
        cv::rectangle(canvas, paint_point, paint_point + cv::Point(template_size, template_size), cv::Scalar(0, 255, 0));

        cv::imshow("template", match_template);
        cv::imshow("search_image", canvas);
        cv::imshow("result_image", result_image / max_val);

        std::cout << "disp = " << point_left.x - min_loc.x - half_template_size << std::endl;
        */
    }
    
    return cv::Point2f(0, 0);
}

std::vector<StereoFeature> StereoFeatureExtractor::extract(
        const cv::Mat& image_left, const cv::Mat& image_right) const
{
    if (match_method_ == KEY_POINT_TO_KEY_POINT)
    {
        return extractKeyPointToKeyPoint(image_left, image_right);
    }
    else
    {
        return extractKeyPointToBlock(image_left, image_right, 100);
    }
}

std::vector<StereoFeature> StereoFeatureExtractor::extractKeyPointToKeyPoint(
        const cv::Mat& image_left, const cv::Mat& image_right) const
{
    assert(feature_extractor_.get() != NULL);
    assert(stereo_camera_model_.get() != NULL);

    std::vector<KeyPoint> key_points_left;
    cv::Mat descriptors_left;

    cv::Rect roi_left = region_of_interest_;
    if (roi_left.x + roi_left.width > image_left.cols) 
        roi_left.width = image_left.cols - roi_left.x;
    if (roi_left.y + roi_left.height > image_left.rows)
        roi_left.height = image_left.rows - roi_left.y;
    
    // std::cout << "roi left: " << roi_left.tl() << " " << roi_left.br() << std::endl;
 
    feature_extractor_->extract(image_left, key_points_left, 
            descriptors_left, roi_left);

    double min_disparity = stereo_camera_model_->getDisparity(max_depth_);
    double max_disparity = stereo_camera_model_->getDisparity(min_depth_);

    std::vector<KeyPoint> key_points_right;
    cv::Mat descriptors_right;

    // compute roi for right image
    cv::Rect roi_right = roi_left;
    if (roi_left.width != 0 && roi_left.height != 0)
    {
        roi_right.x -= max_disparity;
        if (roi_right.x < 0) roi_right.x = 0;
        roi_right.width += max_disparity - min_disparity;
        if (roi_right.width > image_right.cols) roi_right.width = image_right.cols;
    }
    // std::cout << "roi right: " << roi_right.tl() << " " << roi_right.br() << std::endl;

    feature_extractor_->extract(image_right, key_points_right,
            descriptors_right, roi_right);


    cv::Mat match_mask;
    computeMatchMask(key_points_left, key_points_right, match_mask, max_y_diff_,
            max_angle_diff_, max_size_diff_, min_disparity, max_disparity);

    std::vector<cv::DMatch> matches;
    //crossCheckMatching(descriptors_left, descriptors_right, matches, match_mask);
    thresholdMatching(descriptors_left, descriptors_right, matches, match_mask);

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
        stereo_features[i].key_point_left = key_point_left;
        stereo_features[i].key_point_right = key_point_right;
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
        int max_size_diff, double min_disparity, double max_disparity)
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
            if (y_diff <= max_y_diff && disparity > min_disparity && 
                disparity <= max_disparity && angle_diff <= max_angle_diff && 
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

void StereoFeatureExtractor::crossCheckMatching(
                            const cv::Mat& descriptors_left, 
                            const cv::Mat& descriptors_right,
                            std::vector<cv::DMatch>& matches, 
                            const cv::Mat& match_mask)
{
    matches.clear();
    int knn = 2;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = 
        cv::DescriptorMatcher::create("BruteForce");
    std::vector<std::vector<cv::DMatch> > matches_left2right, matches_right2left;
    if (match_mask.empty())
    {
        descriptor_matcher->knnMatch(descriptors_left, descriptors_right,
            matches_left2right, knn);
        descriptor_matcher->knnMatch(descriptors_right, descriptors_left,
            matches_right2left, knn);
    }
    else
    {
        descriptor_matcher->knnMatch(descriptors_left, descriptors_right,
                matches_left2right, knn, match_mask.t());
        descriptor_matcher->knnMatch(descriptors_right, descriptors_left,
                matches_right2left, knn, match_mask);
    }

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
                    matches.push_back(forward);
                    cross_check_found = true;
                    break;
                }
            }
            if (cross_check_found) break;
        }
    }
}

/*
void StereoFeatureExtractor::thresholdMatching(
                            const cv::Mat& descriptors_left, 
                            const cv::Mat& descriptors_right,
                            std::vector<cv::DMatch>& matches, 
                            const cv::Mat& match_mask)
{
    matches.clear();
    int knn = 2;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = 
        cv::DescriptorMatcher::create("BruteForce");
    std::vector<std::vector<cv::DMatch> > matches_left2right;
    if (match_mask.empty())
    {
        descriptor_matcher->knnMatch(descriptors_left, descriptors_right,
                matches_left2right, knn);
    }
    else
    {
        descriptor_matcher->knnMatch(descriptors_left, descriptors_right,
                matches_left2right, knn, match_mask.t());
    }

    for (size_t m = 0; m < matches_left2right.size(); m++ )
    {
        if (matches_left2right[m].size() == 1)
        {
            matches.push_back(matches_left2right[m][0]);
        } 
        else if (matches_left2right[m].size() == 2)
        {
            float dist1 = matches_left2right[m][0].distance;
            float dist2 = matches_left2right[m][1].distance;
            static const float DISTANCE_RATIO_THRESHOLD = 0.8;
            if (dist1 / dist2 < DISTANCE_RATIO_THRESHOLD)
            {
                matches.push_back(matches_left2right[m][0]);
                float new_threshold = (dist2 + dist1) / 2;
                if (dynamic_threshold < 0)
                {
                    dynamic_threshold = new_threshold;
                }
                else
                {
                    dynamic_threshold = 0.99 * dynamic_threshold + 0.01 * new_threshold;
                }
            }
         }
    }
}
*/
void StereoFeatureExtractor::thresholdMatching(
                            const cv::Mat& descriptors_left, 
                            const cv::Mat& descriptors_right,
                            std::vector<cv::DMatch>& matches, 
                            const cv::Mat& match_mask)
{
    matches.clear();
    int knn = 2;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = 
        cv::DescriptorMatcher::create("BruteForce");
    std::vector<std::vector<cv::DMatch> > matches_left2right;
    descriptor_matcher->knnMatch(descriptors_left, descriptors_right,
            matches_left2right, knn);

    for (size_t m = 0; m < matches_left2right.size(); m++ )
    {
        if (matches_left2right[m].size() == 2)
        {
            float dist1 = matches_left2right[m][0].distance;
            float dist2 = matches_left2right[m][1].distance;
            int queryIndex = matches_left2right[m][0].queryIdx;
            int trainIndex = matches_left2right[m][0].trainIdx;
            static const float DISTANCE_RATIO_THRESHOLD = 0.8;
            if (dist1 / dist2 < DISTANCE_RATIO_THRESHOLD &&
                    match_mask.at<unsigned char>(trainIndex, queryIndex) > 0)
            {
                matches.push_back(matches_left2right[m][0]);
            }
        }
    }
}

