#include <opencv2/imgproc/imgproc.hpp>

#include "feature_extraction/smart_surf_descriptor_extractor.h"

feature_extraction::SmartSurfDescriptorExtractor::SmartSurfDescriptorExtractor() : DescriptorExtractor(), 
    smart_surf_()
{
}

void feature_extraction::SmartSurfDescriptorExtractor::extract(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& key_points,
    cv::Mat& descriptors)
{
  if (key_points.size() == 0) return;

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
  smart_surf_.compute(key_points, descriptors);
}

