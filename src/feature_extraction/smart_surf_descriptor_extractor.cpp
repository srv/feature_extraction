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
  smart_surf_.compute(key_points, descriptors);
}

