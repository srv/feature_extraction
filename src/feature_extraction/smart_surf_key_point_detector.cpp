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

  cv::Mat input_image;
  if (!image.isContinuous())
  {
    input_image = image.clone();
  }
  else
  {
    input_image = image;
  }

  cv::Mat gray_image;
  if (input_image.type() == CV_8UC3)
  {
    cv::cvtColor(input_image, gray_image, CV_BGR2GRAY);
  }
  else
  {
    gray_image = input_image;
  }

  smart_surf_.init(gray_image);
  smart_surf_.detect(key_points);
}

