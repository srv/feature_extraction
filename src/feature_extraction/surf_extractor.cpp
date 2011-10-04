#include <opencv2/imgproc/imgproc.hpp>

#include "feature_extraction/surf_extractor.h"

using namespace feature_extraction;

static const int OCTAVES = 5;
static const int INIT_STEP = 2;
static const int THRESHOLD_RESPONSE = 26;

bool keyPointSort(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2)
{
    return kp1.response > kp2.response;
}

SurfExtractor::SurfExtractor() : FeatureExtractor(), 
    surf_(OCTAVES, INIT_STEP, THRESHOLD_RESPONSE)
{
}

void SurfExtractor::extract(const cv::Mat& image,
        std::vector<KeyPoint>& key_points,
        cv::Mat& descriptors, const cv::Rect& roi)
{
    assert(image.type() == CV_8UC3 || image.type() == CV_8U);

    cv::Mat gray_image;
    if (image.type() == CV_8UC3)
    {
        cv::cvtColor(image, gray_image, CV_BGR2GRAY);
    }
    else
    {
        gray_image = image;
    }

    std::vector<cv::KeyPoint> cv_key_points;

    cv::Mat image_region;
    // apply roi
    if (roi.width != 0 && roi.height != 0)
    {
        // we have to copy becaus Surf needs continuous images
        cv::Mat(gray_image, roi).copyTo(image_region);
    }
    else
    {
        image_region = gray_image;
    }

    surf_.init(image_region);
    surf_.detect(cv_key_points);

    if (cv_key_points.size() > (unsigned int)max_num_key_points_)
    {
        std::partial_sort(cv_key_points.begin(), 
                cv_key_points.begin() + max_num_key_points_, 
                cv_key_points.end(), keyPointSort);
        cv_key_points.resize(max_num_key_points_);
    }
    surf_.compute(cv_key_points, descriptors);

    key_points.resize(cv_key_points.size());
    for (size_t i = 0; i < cv_key_points.size(); ++i)
    {
        key_points[i] = KeyPoint(cv_key_points[i]);
        key_points[i].pt.x += roi.x; // adjust position due to roi
        key_points[i].pt.y += roi.y; // adjust position due to roi
    }
}

