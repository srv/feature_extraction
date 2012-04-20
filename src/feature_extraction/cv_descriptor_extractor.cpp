#include <stdexcept>

#include <opencv2/features2d/features2d.hpp>

#include "feature_extraction/cv_descriptor_extractor.h"

feature_extraction::CvDescriptorExtractor::CvDescriptorExtractor(const std::string& name) :
  DescriptorExtractor()
{
  cv_extractor_ = cv::DescriptorExtractor::create(name);
  if (!cv_extractor_)
  {
    throw std::runtime_error("invalid cv extractor " + name + "!");
  }
}

void feature_extraction::CvDescriptorExtractor::extract(const cv::Mat& image,
                                                        std::vector<cv::KeyPoint>& key_points,
                                                        cv::Mat& descriptors)
{
  cv_extractor_->compute(image, key_points, descriptors);
}

