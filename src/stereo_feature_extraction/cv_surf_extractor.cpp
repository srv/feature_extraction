#include <opencv2/imgproc/imgproc.hpp>

#include "cv_surf_extractor.h"

using namespace stereo_feature_extraction;

bool compareKeyPoints(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2)
{
    return kp1.response > kp2.response;
}

CvSurfExtractor::CvSurfExtractor()
{
}

void CvSurfExtractor::extract(const cv::Mat& image,
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

    std::vector<float> descriptor_data;
    std::vector<cv::KeyPoint> cv_key_points;
    cv::Mat mask;
    surf_(gray_image, mask, cv_key_points);

    // filter roi
    if (roi.width != 0 && roi.height != 0)
    {
        std::vector<cv::KeyPoint> filtered_key_points;
        for (size_t i = 0; i < cv_key_points.size(); ++i)
        {
            if (roi.contains(cv_key_points[i].pt))
            {
                filtered_key_points.push_back(cv_key_points[i]);
            }
        }
        cv_key_points = filtered_key_points;
    }

    if (cv_key_points.size() > (unsigned int)max_num_key_points_)
    {
        std::partial_sort(cv_key_points.begin(),
                cv_key_points.begin() + max_num_key_points_,
                cv_key_points.end(),
                compareKeyPoints);
        cv_key_points.resize(max_num_key_points_);
    }
    bool use_provided_key_points = true;
    surf_(gray_image, mask, cv_key_points, descriptor_data, use_provided_key_points);

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
