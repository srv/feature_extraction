#include <algorithm>
#include <cassert>

#include "feature_extraction/key_points_filter.h"

bool feature_extraction::KeyPointsFilter::responseCompare(
    const cv::KeyPoint& kp1,
    const cv::KeyPoint& kp2)
{
  return kp1.response > kp2.response;
}

void feature_extraction::KeyPointsFilter::filterBest(
    std::vector<cv::KeyPoint>& key_points, int max_num)
{
  assert(max_num >= 0);
  if (key_points.size() <= static_cast<unsigned int>(max_num)) return;
  std::partial_sort(key_points.begin(), key_points.begin() + max_num,
                    key_points.end(), responseCompare);
  key_points.resize(max_num);
}
