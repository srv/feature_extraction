#ifndef KEY_POINT_H
#define KEY_POINT_H

#include <opencv2/features2d/features2d.hpp>


namespace feature_extraction
{
/**
* KeyPoint class. For now we just inherit from cv::KeyPoint
*/
class KeyPoint : public cv::KeyPoint
{
  public:

      KeyPoint(const cv::KeyPoint& key_point) : cv::KeyPoint(key_point) {}
      KeyPoint() : cv::KeyPoint() {}
};

}

#endif

