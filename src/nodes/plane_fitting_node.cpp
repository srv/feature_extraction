#include <ros/ros.h>

#include <geometry_msgs/PointStamped.h>

#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_ros/point_cloud.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class PlaneFittingNode
{
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Subscriber point_cloud_sub_;
  ros::Publisher mean_dist_pub_;
  ros::Publisher plane_dist_pub_;

  // minimum number of inliers for robust plane estimation
  int min_num_inliers_;
  // threshold for ransac plane fitting inliers
  double inlier_threshold_;

public:
  PlaneFittingNode() : nh_private_("~")
  {
    init();
  }

  ~PlaneFittingNode()
  {
  }

  void init()
  {
    point_cloud_sub_ = nh_.subscribe<PointCloud>("point_cloud", 1, &PlaneFittingNode::pointCloudCb, this);
    mean_dist_pub_ = nh_private_.advertise<geometry_msgs::PointStamped>("mean_distance", 1);
    plane_dist_pub_ = nh_private_.advertise<geometry_msgs::PointStamped>("plane_distance", 1);

    nh_private_.param("min_num_inliers", min_num_inliers_, 10);
    if (min_num_inliers_ < 3)
    {
      ROS_WARN("Given minimum number of inliers must be > 2, setting to 3.");
      min_num_inliers_ = 3;
    } 

    nh_private_.param("inlier_threshold", inlier_threshold_, 0.1);
    ROS_INFO_STREAM("Parameters: \n"
                    "  min_num_inliers = " << min_num_inliers_ << "\n"
                    "  inlier_threshold = " << inlier_threshold_);
  }

  void pointCloudCb(const PointCloud::ConstPtr& point_cloud)
  {
    publishMeanDistance(point_cloud);

    geometry_msgs::PointStamped plane_dist_msg;
    plane_dist_msg.header = point_cloud->header;
    if (point_cloud->points.size() > static_cast<unsigned int>(min_num_inliers_))
    {
      pcl::ModelCoefficients coefficients;
      pcl::PointIndices inliers;
      // Create the segmentation object
      pcl::SACSegmentation<PointCloud::PointType> seg;
      // Optional
      seg.setOptimizeCoefficients (true);
      // Mandatory
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setDistanceThreshold(inlier_threshold_); 
      seg.setInputCloud(point_cloud);
      seg.segment(inliers, coefficients);

      // there must be some inliers
      if(inliers.indices.size() < static_cast<unsigned int>(min_num_inliers_))
      {
        ROS_ERROR("not enough inliers!");
        return;
      }

      ROS_INFO_STREAM("plane model inliers: " << inliers.indices.size());
      double a = coefficients.values[0];
      double b = coefficients.values[1];
      double c = coefficients.values[2];
      double d = coefficients.values[3];

      double dist = d / sqrt(a*a + b*b + c*c);
      plane_dist_msg.point.z = dist > 0 ? dist : -dist;

      ROS_INFO_STREAM("fitted plane coefficients: " << a << " " << b << " "
                                                    << c << " " << d << " (angle to z axis: " << acos(c) / M_PI * 180.0  << ")");

      std::vector<double> distances(inliers.indices.size());
      for (size_t i = 0; i < inliers.indices.size(); ++i)
      {
        const PointCloud::PointType& point = point_cloud->points[inliers.indices[i]];
        distances[i] = std::abs(a * point.x + b * point.y + c * point.z + d);
      }

      // http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
      double m2 = 0;     // helper for floating variance computation
      double mean_distance = 0;
      double min_distance = std::numeric_limits<double>::max();
      double max_distance = 0;
      for (size_t i = 0; i < distances.size(); ++i)
      {
        if (distances[i] > max_distance) max_distance = distances[i];
        if (distances[i] < min_distance) min_distance = distances[i];
        double delta = distances[i] - mean_distance;
        mean_distance += delta / (i + 1);
        m2 += delta * (distances[i] - mean_distance);
      }
      double variance = m2 / (distances.size() - 1);
      ROS_INFO_STREAM("inlier distances from fitted plane: mean = " << mean_distance
                                                                    << " stddev = " << sqrt(variance) << " min = " << min_distance
                                                                    << " max = " << max_distance);
    }
    else
    {
      ROS_INFO("Not enough points to compute plane.");
      plane_dist_msg.point.z = -1;
    }
    plane_dist_pub_.publish(plane_dist_msg);
  }

  void publishMeanDistance(const PointCloud::ConstPtr& point_cloud)
  {
    geometry_msgs::PointStamped dist_msg;
    dist_msg.header = point_cloud->header;
    if (point_cloud->points.size() == 0)
    {
      dist_msg.point.z = -1;
    }
    else
    {
      double mean_dist, min_dist, max_dist, mean_z, min_z, max_z;
      mean_dist = mean_z = 0.0;
      min_dist = min_z = std::numeric_limits<double>::max();
      max_dist = max_z = 0.0;
      for (size_t i = 0; i < point_cloud->points.size(); ++i)
      {
        const PointCloud::PointType& point = point_cloud->points[i];
        double dist = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        mean_dist += dist;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
        mean_z += point.z;
        if (point.z < min_z) min_z = point.z;
        if (point.z > max_z) max_z = point.z;
      }
      mean_dist /= point_cloud->points.size();
      mean_z /= point_cloud->points.size();

      ROS_INFO_STREAM("Distances to origin: MIN: " << min_dist << "\tMAX: " << max_dist << " \tMEAN: " << mean_dist);
      ROS_INFO_STREAM("                  Z: MIN: " << min_z << "\tMAX: " << max_z << " \tMEAN: " << mean_z);

      dist_msg.point.z = mean_z;
    }
    mean_dist_pub_.publish(dist_msg);
  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "plane_fitting");
  PlaneFittingNode node;
  ros::spin();
  return 0;
}

