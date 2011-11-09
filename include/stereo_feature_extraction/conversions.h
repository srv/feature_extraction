#include <opencv2/features2d/features2d.hpp>
#include <vision_msgs/Features3D.h>

namespace stereo_feature_extraction
{
  inline void fromMsg(const vision_msgs::KeyPoint& kp, cv::KeyPoint& cv_kp)
  {
    cv_kp.pt.x = kp.x;
    cv_kp.pt.y = kp.y;
    cv_kp.size = kp.size;
    cv_kp.angle = kp.angle;
    cv_kp.response = kp.response;
    cv_kp.octave = kp.octave;
    // cv::KeyPoint has no field for laplacian
  }

  inline void toMsg(const cv::KeyPoint& cv_kp, vision_msgs::KeyPoint& kp)
  {
    kp.x = cv_kp.pt.x;
    kp.y = cv_kp.pt.y;
    kp.size = cv_kp.size;
    kp.angle = cv_kp.angle;
    kp.response = cv_kp.response;
    kp.octave = cv_kp.octave;
    // cv::KeyPoint has no field for laplacian
  }

  inline void fromMsg(const vision_msgs::Features& features_msg, std::vector<cv::KeyPoint>& key_points, cv::Mat& descriptors)
  {
    key_points.resize(features_msg.key_points.size());
    for (size_t i = 0; i < key_points.size(); ++i)
    {
      fromMsg(features_msg.key_points[i], key_points[i]);
    }
    size_t descriptor_length = features_msg.descriptor_data.size() / features_msg.key_points.size();
    descriptors = cv::Mat(key_points.size(), descriptor_length, features_msg.descriptor_data[0]).clone();
  }

  inline void toMsg(const std::vector<cv::KeyPoint>& key_points, const cv::Mat& descriptors, vision_msgs::Features& features_msg)
  {
    assert(descriptors.isContinuous());
    assert(descriptors.depth() == CV_32F);
    assert(descriptors.channels() == 1);

    features_msg.key_points.resize(key_points.size());
    for (size_t i = 0; i < features_msg.key_points.size(); ++i)
    {
      toMsg(key_points[i], features_msg.key_points[i]);
    }
    size_t data_length = descriptors.rows * descriptors.cols;
    features_msg.descriptor_data.resize(data_length);
    std::copy(descriptors.ptr(), descriptors.ptr() + data_length, features_msg.descriptor_data.begin());
   }
}

