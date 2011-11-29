#ifndef SMART_SURF_KEY_POINT_DETECTOR_H
#define SMART_SURF_KEY_POINT_DETECTOR_H

#include "feature_extraction/key_point_detector.h"

#include "feature_extraction/smart_surf.h"

namespace feature_extraction
{

/**
 * \class SmartSurfKeyPointDetector
 * \brief Key Point Detector for SmartSURF
 */
class SmartSurfKeyPointDetector : public KeyPointDetector
{

public:

  SmartSurfKeyPointDetector();

  virtual ~SmartSurfKeyPointDetector() {};

  /**
   * Detection interface.
   * \param image the input image.
   * \param key_points output vector for computed key points
   * \param roi region of interest where the key points have to be extracted
   */
  virtual void detect(const cv::Mat& image,
                       std::vector<cv::KeyPoint>& key_points);

protected:

  visual_odometry::SmartSurf smart_surf_;
};

}

#endif


