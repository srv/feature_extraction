#include <stdexcept>

#include <opencv2/features2d/features2d.hpp>

#include "feature_extraction/cv_key_point_detector.h"

feature_extraction::CvKeyPointDetector::CvKeyPointDetector(const std::string& name) :
  KeyPointDetector()
{
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

