#include <opencv2/imgproc/imgproc.hpp>

#include "surf_extractor.h"

using namespace stereo_feature_extraction;

static const int OCTAVES = 5;
static const int INIT_STEP = 2;
static const int THRESHOLD_RESPONSE = 26;
static const int MAX_NUM_POINTS = 200;

SurfExtractor::SurfExtractor() :
    surf_(OCTAVES, INIT_STEP, THRESHOLD_RESPONSE)
{
}

void SurfExtractor::extract(const cv::Mat& image, const cv::Mat& mask,
        std::vector<KeyPoint>& key_points,
        cv::Mat& descriptors)
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
    
    surf_.init(gray_image);
    surf_.detect(cv_key_points);
    if (cv_key_points.size() > MAX_NUM_POINTS)
    {
        cv_key_points.resize(MAX_NUM_POINTS);
    }
    surf_.compute(cv_key_points, descriptors);

    key_points.resize(cv_key_points.size());
    for (size_t i = 0; i < cv_key_points.size(); ++i)
    {
        key_points[i] = KeyPoint(cv_key_points[i]);
    }
}
