#include <opencv2/imgproc/imgproc.hpp>

#include "cv_surf_extractor.h"

using namespace stereo_feature_extraction;

CvSurfExtractor::CvSurfExtractor()
{
}

void CvSurfExtractor::extract(const cv::Mat& image, const cv::Mat& mask,
        std::vector<KeyPoint>& key_points,
        cv::Mat& descriptors) const
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

    std::vector<float> descriptor_data;
    std::vector<cv::KeyPoint> cv_key_points;
    surf_(gray_image, mask, cv_key_points, descriptor_data);

    key_points.resize(cv_key_points.size());
    for (size_t i = 0; i < cv_key_points.size(); ++i)
    {
        key_points[i] = KeyPoint(cv_key_points[i]);
    }

    // copy to output matrix
    bool copyData = true;
    descriptors = 
        cv::Mat(descriptor_data, copyData).reshape(1, key_points.size());
}
