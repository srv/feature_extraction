#ifndef KEY_POINTS_FILTER_H
#define KEY_POINTS_FILTER_H

#include <vector>

#include <opencv2/features2d/features2d.hpp>

namespace feature_extraction
{

/**
 * \brief Filter for key points
 */
class KeyPointsFilter
{

public:
  /**
   * Filters given key_points in place so that the strongest max_num survive.
   */
  static void filterBest(std::vector<cv::KeyPoint>& key_points, int max_num);


  /**
  * For sorting key points by response, biggest response first.
  */
  static bool responseCompare(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2);

};

}

#endif


