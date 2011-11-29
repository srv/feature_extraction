#include <stdexcept>

#include <opencv2/features2d/features2d.hpp>

#include "feature_extraction/smart_surf_key_point_detector.h"

feature_extraction::SmartSurfKeyPointDetector::SmartSurfKeyPointDetector() :
  KeyPointDetector(), smart_surf_()
{
}

void feature_extraction::SmartSurfKeyPointDetector::detect(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& key_points)
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

  smart_surf_.init(gray_image);
  smart_surf_.detect(key_points);
}

