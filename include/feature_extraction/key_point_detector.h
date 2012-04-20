#ifndef KEY_POINT_DETECTOR_H
#define KEY_POINT_DETECTOR_H

#include <vector>
#include <boost/shared_ptr.hpp>

namespace cv
{

class KeyPoint;
class Mat;

}

namespace feature_extraction
{

/**
 * \class KeyPointDetector
 * \brief Common interface for key point detectors
 */
class KeyPointDetector
{

public:

  /**
   * Default constructor
   */
  KeyPointDetector() {}

  /**
   * Virtual destructor
   */
  virtual ~KeyPointDetector() {}

  /**
   * Detection interface.
   * \param image the input image.
   * \param key_points output vector for computed key points
   */
  virtual void detect(const cv::Mat& image,
                       std::vector<cv::KeyPoint>& key_points) = 0;

  typedef boost::shared_ptr<KeyPointDetector> Ptr;

};

}

#endif


