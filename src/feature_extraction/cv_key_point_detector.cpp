#include <stdexcept>

#include <opencv2/features2d/features2d.hpp>

#if (CV_MAJOR_VERSION > 2 ) || ( CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >= 4)
  #include <opencv2/nonfree/nonfree.hpp>
#endif

#include "feature_extraction/cv_key_point_detector.h"


bool feature_extraction::CvKeyPointDetector::nonfree_module_initialized_ = false;

feature_extraction::CvKeyPointDetector::CvKeyPointDetector(const std::string& name) :
  KeyPointDetector()
{

#if (CV_MAJOR_VERSION > 2 ) || ( CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >= 4)
  if (!nonfree_module_initialized_)
  {
    cv::initModule_nonfree();
    nonfree_module_initialized_ = true;
  }
#endif
  cv_detector_ = cv::FeatureDetector::create(name);
  if (!cv_detector_)
  {
    throw std::runtime_error("invalid cv detector " + name + "!");
  }
}

void feature_extraction::CvKeyPointDetector::detect(const cv::Mat& image,
                                                    std::vector<cv::KeyPoint>& key_points)
{
  cv_detector_->detect(image, key_points);
}

