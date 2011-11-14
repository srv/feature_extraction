
#include <iostream>
#include "feature_matching/matching_constraints.h"

void feature_matching::matching_constraints::computeStereoMatchMask(
        const std::vector<cv::KeyPoint>& key_points_left,
        const std::vector<cv::KeyPoint>& key_points_right,
        cv::Mat& match_mask, double max_y_diff, double max_angle_diff, 
        int max_size_diff, double min_disparity, double max_disparity)
{
  if (key_points_left.empty() || key_points_right.empty())
  {
    return;
  }

  int y_diff_ok, angle_diff_ok, disparity_ok, size_diff_ok;
  y_diff_ok = angle_diff_ok = disparity_ok = size_diff_ok = 0;

  match_mask.create(key_points_left.size(), key_points_right.size(), CV_8UC1);
  for (int r = 0; r < match_mask.rows; ++r)
  {
    const cv::KeyPoint& keypoint_left = key_points_left[r];
    for (int c = 0; c < match_mask.cols; ++c)
    {
      const cv::KeyPoint& keypoint_right = key_points_right[c];
      bool allow_match = false;
      // y_diff check, filters out most mismatches
      if (fabs(keypoint_left.pt.y - keypoint_right.pt.y) <= max_y_diff)
      {
        y_diff_ok++;
        // angle check
        // NOTE: cv::KeyPoint carries angle information in degrees
        double angle_diff = std::abs(keypoint_left.angle - keypoint_right.angle);
        angle_diff = std::min(360 - angle_diff, angle_diff); 
        if (angle_diff <= max_angle_diff)
        {
          angle_diff_ok++;
          // disparity check
          double disparity = keypoint_left.pt.x - keypoint_right.pt.x;
          if (disparity >= min_disparity && disparity <= max_disparity)
          {
            disparity_ok++;
            // size check
            if (std::abs(keypoint_left.size - keypoint_right.size) <= max_size_diff)
            {
              size_diff_ok++;
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
  /*
  std::cout << match_mask.rows * match_mask.cols << " possible matches: " << std::endl
            << "  " << y_diff_ok     << " y diff <= " << max_y_diff << " ok " << std::endl
            << "    " << angle_diff_ok << " angle diff <= " << max_angle_diff << " ok " << std::endl
            << "      " << disparity_ok  << " disparity >= " << min_disparity << " && disparity <= " << max_disparity << " ok " << std::endl
            << "        " << size_diff_ok  << " size diff <= " << max_size_diff << " ok " << std::endl;
            */
}

