#include <opencv2/features2d/features2d.hpp>
#include <vision_msgs/Features3D.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace feature_extraction_ros
{

inline void fromMsg(const vision_msgs::KeyPoint& kp_msg, cv::KeyPoint& kp)
{
  kp.pt.x = kp_msg.x;
  kp.pt.y = kp_msg.y;
  kp.size = kp_msg.size;
  kp.angle = kp_msg.angle;
  kp.response = kp_msg.response;
  kp.octave = kp_msg.octave;
}

inline void toMsg(const cv::KeyPoint& kp, vision_msgs::KeyPoint& kp_msg)
{
  kp_msg.x = kp.pt.x;
  kp_msg.y = kp.pt.y;
  kp_msg.size = kp.size;
  kp_msg.angle = kp.angle;
  kp_msg.response = kp.response;
  kp_msg.octave = kp.octave;
}

inline void fromMsg(const vision_msgs::Mat& mat_msg, cv::Mat& mat)
{
  mat.create(mat_msg.rows, mat_msg.cols, mat_msg.type);
  std::copy(mat_msg.data.begin(), mat_msg.data.end(), mat.data);
}

inline void toMsg(const cv::Mat& mat, vision_msgs::Mat& mat_msg)
{
  assert(mat.channels() == 1);
  assert(mat.dims == 2);
  assert(mat.isContinuous());
  mat_msg.rows = mat.rows;
  mat_msg.cols = mat.cols;
  mat_msg.type = mat.type();
  int num_bytes = mat.rows * mat.cols * mat.elemSize();
  mat_msg.data.resize(num_bytes);
  std::copy(mat.data, mat.data + num_bytes, mat_msg.data.begin());
}

inline void fromMsg(const vision_msgs::Features& features_msg, 
                    std::vector<cv::KeyPoint>& key_points, cv::Mat& descriptors)
{
  if (features_msg.key_points.size() == 0) return;

  key_points.resize(features_msg.key_points.size());
  for (size_t i = 0; i < key_points.size(); ++i)
  {
    fromMsg(features_msg.key_points[i], key_points[i]);
  }

  fromMsg(features_msg.descriptor_data, descriptors);
}

inline void toMsg(const std::vector<cv::KeyPoint>& key_points, 
                  const cv::Mat& descriptors, 
                  vision_msgs::Features& features_msg)
{
  if (key_points.size() == 0) return;

  features_msg.key_points.resize(key_points.size());
  for (size_t i = 0; i < features_msg.key_points.size(); ++i)
  {
    toMsg(key_points[i], features_msg.key_points[i]);
  }

  toMsg(descriptors, features_msg.descriptor_data);
}

inline void toMsg(const std::vector<cv::Point3d> world_points, 
                  pcl::PointCloud<pcl::PointXYZ>& point_cloud)
{
  point_cloud.points.resize(world_points.size());
  point_cloud.height = 1;
  point_cloud.width = world_points.size();
  for (size_t i = 0; i < world_points.size(); ++i)
  {
    point_cloud.points[i].x = world_points[i].x;
    point_cloud.points[i].y = world_points[i].y;
    point_cloud.points[i].z = world_points[i].z;
  }
}

inline void toMsg(const std::vector<cv::KeyPoint> key_points, 
                  const cv::Mat& descriptors,
                  const std::vector<cv::Point3d> world_points, 
                  vision_msgs::Features3D& features3d_msg)
{
  assert(key_points.size() == world_points.size());
  assert(key_points.size() == static_cast<unsigned int>(descriptors.rows));
  toMsg(key_points, descriptors, features3d_msg.features_left);
  features3d_msg.world_points.resize(world_points.size());
  for (size_t i = 0; i < features3d_msg.world_points.size(); ++i)
  {
    features3d_msg.world_points[i].x = world_points[i].x;
    features3d_msg.world_points[i].y = world_points[i].y;
    features3d_msg.world_points[i].z = world_points[i].z;
  }
}

}

