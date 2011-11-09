
#include "stereo_matching.h"

void stereo_feature_extraction::stereo_matching:::computeMatchMask(
        const std::vector<cv::KeyPoint>& key_points_left,
        const std::vector<cv::KeyPoint>& key_points_right,
        cv::Mat& match_mask, double max_y_diff, double max_angle_diff, 
        int max_size_diff, double min_disparity, double max_disparity)
{
  if (key_points_left.empty() || key_points_right.empty())
  {
    return;
  }

  match_mask.create(key_points_right.size(), key_points_left.size(), CV_8UC1);
  for (int r = 0; r < match_mask.rows; ++r)
  {
    const KeyPoint& keypoint2 = key_points_right[r];
    for (int c = 0; c < match_mask.cols; ++c)
    {
      const KeyPoint& keypoint1 = key_points_left[c];
      bool allow_match = false;
      // y_diff check, filters out most mismatches
      if (fabs(keypoint1.pt.y - keypoint2.pt.y) <= max_y_diff)
      {
        // angle check
        // NOTE: cv::KeyPoint carries angle information in degrees
        double angle_diff = std::abs(keypoint1.angle - keypoint2.angle);
        angle_diff = std::min(360 - angle_diff, angle_diff); 
        if (angle_diff <= max_angle_diff)
        {
          // disparity check
          double disparity = keypoint1.pt.x - keypoint2.pt.x;
          if (disparity >= min_disparity && disparity <= max_disparity)
          {
            // size check
            if (std::abs(keypoint1.size - keypoint2.size) <= max_size_diff)
            {
              allow_match = true;
            }
          }
        }
      }

      if (allow_match)
      {
        match_mask.at<unsigned char>(r, c) = 255;
      }
      else
      {
        match_mask.at<unsigned char>(r, c) = 0;
      }
    }
  }
}


void stereo_feature_extraction::stereo_matching::thresholdMatching(
                            const cv::Mat& descriptors1, 
                            const cv::Mat& descriptors2,
                            std::vector<cv::DMatch>& matches, 
                            double threshold,
                            const cv::Mat& match_mask)
{
  matches.clear();
  int knn = 2;
  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = 
      cv::DescriptorMatcher::create("BruteForce");
  std::vector<std::vector<cv::DMatch> > matches_1to2;
  descriptor_matcher->knnMatch(descriptors1, descriptors2,
          matches_1to2, knn);

  for (size_t m = 0; m < matches_1to2.size(); m++ )
  {
    if (matches_1to2.size() == 2) // this should always be the case
    {
      float dist1 = matches_1to2[m][0].distance;
      float dist2 = matches_1to2[m][1].distance;
      int queryIndex = matches_1to2[m][0].queryIdx;
      int trainIndex = matches_1to2[m][0].trainIdx;
      if (dist1 / dist2 < threshold && match_mask.at<unsigned char>(trainIndex, queryIndex) > 0)
      {
        matches.push_back(matches_1to2[m][0]);
      }
    }
  }
}

