#include <stdexcept>

#include <opencv2/features2d/features2d.hpp>

#if (CV_MAJOR_VERSION > 2 ) || ( CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >= 4)
#include <opencv2/nonfree/nonfree.hpp>
#endif

#include "feature_extraction/cv_descriptor_extractor.h"

bool feature_extraction::CvDescriptorExtractor::nonfree_module_initialized_ = false;

feature_extraction::CvDescriptorExtractor::CvDescriptorExtractor(const std::string& name) :
  DescriptorExtractor()
{
#if (CV_MAJOR_VERSION > 2 ) || ( CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >= 4)
  if (!nonfree_module_initialized_)
  {
    cv::initModule_nonfree();
    nonfree_module_initialized_ = true;
  }
#endif
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

