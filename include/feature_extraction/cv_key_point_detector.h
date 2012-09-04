#ifndef CV_KEY_POINT_DETECTOR_H
#define CV_KEY_POINT_DETECTOR_H

#include <opencv2/features2d/features2d.hpp>

#include "key_point_detector.h"

namespace feature_extraction
{

/**
 * \class CvKeyPointDetector
 * \brief Key Point Detector for OpenCV detectors
 */
class CvKeyPointDetector : public KeyPointDetector
{

public:

  /**
   * \param detector name, as given in 
   * http://opencv.itseez.com/modules/features2d/doc/common_interfaces_of_feature_detectors.html#featuredetector-create
   * Throws an exception if name was invalid.
   */
  CvKeyPointDetector(const std::string& name);

  virtual ~CvKeyPointDetector() {};

  /**
   * Detection interface.
   * \param image the input image.
   * \param key_points output vector for computed key points
   * \param roi region of interest where the key points have to be extracted
   */
  virtual void detect(const cv::Mat& image,
                       std::vector<cv::KeyPoint>& key_points);

protected:

  cv::Ptr<cv::FeatureDetector> cv_detector_;

  static bool nonfree_module_initialized_;

};

}

#endif


