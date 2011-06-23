#include <ros/ros.h>

#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <vision_msgs/StereoFeatures.h>

#include "stereo_feature_extractor.h"
#include "feature_extractor_factory.h"
#include "drawing.h"

using namespace stereo_feature_extraction;

class PlaneFittingNode
{
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    ros::Subscriber stereo_features_sub_;

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
        stereo_features_sub_ = nh_.subscribe("stereo_features", 1, &PlaneFittingNode::stereoFeaturesCb, this);
    }

    void stereoFeaturesCb(const vision_msgs::StereoFeaturesConstPtr& features_msg)
    {
        // return if there are no features
        if (features_msg->world_points.width * features_msg->world_points.height == 0) 
        {
            return;
        }

        pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
        pcl::fromROSMsg(features_msg->world_points, point_cloud);

        pcl::ModelCoefficients coefficients;
        pcl::PointIndices inliers;
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        // Optional
        seg.setOptimizeCoefficients (true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.05);
        seg.setInputCloud(point_cloud.makeShared());
        seg.segment(inliers, coefficients);

        // there must be some inliers
        if(inliers.indices.size() < 3)
        {
            ROS_ERROR("not enough inliers!");
            return;
        }

        std::cout << "plane model inliers: " << inliers.indices.size () << std::endl;
        double a = coefficients.values[0];
        double b = coefficients.values[1];
        double c = coefficients.values[2];
        double d = coefficients.values[3];

        std::cout << "fitted plane coefficients: " << a << " " << b << " " 
            << c << " " << d << std::endl;

        std::vector<double> distances(inliers.indices.size());
        for (size_t i = 0; i < inliers.indices.size(); ++i)
        {
            const pcl::PointXYZRGB& point = point_cloud.points[inliers.indices[i]];
            distances[i] = std::abs(a * point.x + b * point.y + c * point.z + d);
        }

        // http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        double m2 = 0; // helper for floating variance computation
        double mean_distance = 0;
        double min_distance = numeric_limits<double>::max();
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
        std::cout << "distances from fitted plane: mean = " << mean_distance
            << " stddev = " << sqrt(variance) << " min = " << min_distance 
            << " max = " << max_distance << std::endl;

        cv::Mat canvas;
    }
 
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "plane_fitting");
    PlaneFittingNode node;
    ros::spin();
    return 0;
}

