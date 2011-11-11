#include <ros/ros.h>

#include <std_msgs/Float64.h>

#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_ros/point_cloud.h>

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

class PlaneFittingNode
{
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    ros::Subscriber point_cloud_sub_;
    ros::Publisher mean_dist_pub_;

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
        mean_dist_pub_ = nh_.advertise<std_msgs::Float64>("mean_distance", 1);
    }

    void pointCloudCb(const PointCloud::ConstPtr& point_cloud)
    {
        publishMeanDistance(point_cloud);

        if (point_cloud->points.size() > 3)
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
          seg.setDistanceThreshold(0.25);
          seg.setInputCloud(point_cloud);
          seg.segment(inliers, coefficients);

          // there must be some inliers
          if(inliers.indices.size() < 3)
          {
              ROS_ERROR("not enough inliers!");
              return;
          }

          ROS_INFO_STREAM("plane model inliers: " << inliers.indices.size());
          double a = coefficients.values[0];
          double b = coefficients.values[1];
          double c = coefficients.values[2];
          double d = coefficients.values[3];

          ROS_INFO_STREAM("fitted plane coefficients: " << a << " " << b << " " 
              << c << " " << d << " (angle to z axis: " << acos(c) / M_PI * 180.0  << ")");

          std::vector<double> distances(inliers.indices.size());
          for (size_t i = 0; i < inliers.indices.size(); ++i)
          {
              const PointCloud::PointType& point = point_cloud->points[inliers.indices[i]];
              distances[i] = std::abs(a * point.x + b * point.y + c * point.z + d);
          }

          // http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
          double m2 = 0; // helper for floating variance computation
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
          ROS_INFO_STREAM("distances from fitted plane: mean = " << mean_distance
              << " stddev = " << sqrt(variance) << " min = " << min_distance 
              << " max = " << max_distance);
        }
        else
        {
          ROS_INFO("Not enough points to compute plane.");
        }
    }

    void publishMeanDistance(const PointCloud::ConstPtr& point_cloud)
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

        std_msgs::Float64 dist_msg;
        dist_msg.data = mean_z;

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

