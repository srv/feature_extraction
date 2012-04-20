#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <vision_msgs/Features3D.h>
#include <sensor_msgs/CameraInfo.h>

#include "feature_extraction_ros/conversions.h"
#include "feature_matching/stereo_feature_matcher.h"
#include "feature_matching/stereo_depth_estimator.h"

namespace feature_extraction_ros
{

class StereoFeatureMatcherNodelet : public nodelet::Nodelet
{
  public:

    StereoFeatureMatcherNodelet() { }

    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

    virtual void onInit()
    {
      ros::NodeHandle nh = getNodeHandle();
      ros::NodeHandle& nh_private = getPrivateNodeHandle();
      subscribed_ = false; // no subscription yet
      
      ros::SubscriberStatusCallback connect_cb = 
          boost::bind(&StereoFeatureMatcherNodelet::connectCb, this);

      pub_point_cloud_ = 
        nh_private.advertise<PointCloud>("point_cloud", 1, connect_cb, connect_cb);

      pub_features_3d_ = 
        nh_private.advertise<vision_msgs::Features3D>("features_3d", 1, connect_cb, connect_cb);

      int queue_size;
      nh_private.param("queue_size", queue_size, 5);

      nh_private.param("max_y_diff", max_y_diff_, 2.0);
      nh_private.param("max_angle_diff", max_angle_diff_, 2.0);
      nh_private.param("max_size_diff", max_size_diff_, 2);
      nh_private.param("matching_threshold", matching_threshold_, 0.8);

      NODELET_INFO_STREAM("Parameters: \n"
                          " max_y_diff = " << max_y_diff_  << "\n"
                          " max_angle_diff = " << max_angle_diff_ << "\n"
                          " max_size_diff = " << max_size_diff_ << "\n"
                          " matching_threshold = " << matching_threshold_);

      // Synchronize inputs. Topic subscriptions happen on demand in the 
      // connection callback.
      exact_sync_.reset(new ExactSync(ExactPolicy(queue_size), sub_l_features_, sub_r_features_, sub_l_info_, sub_r_info_));
      exact_sync_->registerCallback(boost::bind(&StereoFeatureMatcherNodelet::featuresCb, this, _1, _2, _3, _4));

      NODELET_INFO("Waiting for client subscriptions.");
    }

    // Handles (un)subscribing when clients (un)subscribe
    void connectCb()
    {
      if (pub_features_3d_.getNumSubscribers() == 0 &&
          pub_point_cloud_.getNumSubscribers() == 0)
      {
        NODELET_INFO("No more clients connected, unsubscribing from inputs.");
        sub_l_features_  .unsubscribe();
        sub_r_features_  .unsubscribe();
        sub_l_info_   .unsubscribe();
        sub_r_info_   .unsubscribe();
        subscribed_ = false;
      }
      else if (!subscribed_)
      {
        NODELET_INFO("Client connected, subscribing to inputs.");
        ros::NodeHandle nh = getNodeHandle();
        // Queue size 1 should be OK; the one that matters is the synchronizer queue size.
        sub_l_features_.subscribe(nh, "features_left", 1);
        sub_r_features_.subscribe(nh, "features_right", 1);
        sub_l_info_    .subscribe(nh, "camera_info_left", 1);
        sub_r_info_    .subscribe(nh, "camera_info_right", 1);
        subscribed_ = true;
      }
    }

    void featuresCb(const vision_msgs::FeaturesConstPtr& l_features_msg,
                 const vision_msgs::FeaturesConstPtr& r_features_msg,
                 const sensor_msgs::CameraInfoConstPtr& l_info_msg,
                 const sensor_msgs::CameraInfoConstPtr& r_info_msg)
    {
      ros::WallTime start_time = ros::WallTime::now();

      // convert from msg to native formats
      std::vector<cv::KeyPoint> key_points_left;
      cv::Mat descriptors_left;
      feature_extraction_ros::fromMsg(*l_features_msg, key_points_left, descriptors_left);

      std::vector<cv::KeyPoint> key_points_right;
      cv::Mat descriptors_right;
      feature_extraction_ros::fromMsg(*r_features_msg, key_points_right, descriptors_right);

      // configure and perform matching
      feature_matching::StereoFeatureMatcher::Params params;
      params.max_y_diff = max_y_diff_;
      params.max_angle_diff = max_angle_diff_;
      params.max_size_diff = max_size_diff_;

      feature_matching::StereoFeatureMatcher matcher;
      matcher.setParams(params);
      std::vector<cv::DMatch> matches;
      matcher.match(key_points_left, descriptors_left, key_points_right, 
          descriptors_right, matching_threshold_, matches);

      // calculate 3D world points
      feature_matching::StereoDepthEstimator depth_estimator;
      depth_estimator.setCameraInfo(*l_info_msg, *r_info_msg);
      std::vector<cv::Point3d> world_points(matches.size());
      for (size_t i = 0; i < matches.size(); ++i)
      {
        depth_estimator.calculate3DPoint(key_points_left[matches[i].queryIdx].pt,
                                         key_points_right[matches[i].trainIdx].pt,
                                         world_points[i]);
      }

      // convert to msg format again
      PointCloud::Ptr point_cloud(new PointCloud());
      feature_extraction_ros::toMsg(world_points, *point_cloud);
      point_cloud->header = l_features_msg->header;

      vision_msgs::Features3D::Ptr features_3d_msg(new vision_msgs::Features3D());
      feature_extraction_ros::toMsg(key_points_left, descriptors_left, 
                                    world_points, *features_3d_msg);
      features_3d_msg->header = l_features_msg->header;
      
      /*

      point_cloud->points.resize(matches.size());
      point_cloud->header = l_info_msg->header;

      size_t descriptor_length = l_features_msg->descriptor_data.size() / l_features_msg->key_points.size();
      
      vision_msgs::Features3D::Ptr features_3d_msg(new vision_msgs::Features3D());
      features_3d_msg->world_points.resize(matches.size());
      features_3d_msg->features_left.key_points.resize(matches.size());
      features_3d_msg->features_left.descriptor_data.resize(descriptor_length * matches.size());
      features_3d_msg->features_left.descriptor_name = l_features_msg->descriptor_name;
      features_3d_msg->features_left.header = l_features_msg->header;
      features_3d_msg->header = l_info_msg->header;


      for (size_t i = 0; i < matches.size(); ++i)
      {
        int index_left = matches[i].queryIdx;
        int index_right = matches[i].trainIdx;
        const cv::KeyPoint& key_point_left = key_points_left[index_left];
        const cv::KeyPoint& key_point_right = key_points_right[index_right];
        float disparity = key_point_left.pt.x - key_point_right.pt.x;
        cv::Point3d world_point;
        stereo_camera_model_.projectDisparityTo3d(key_point_left.pt, disparity, world_point);
        point_cloud->points[i].x = world_point.x;
        point_cloud->points[i].y = world_point.y;
        point_cloud->points[i].z = world_point.z;

        feature_extraction_ros::toMsg(key_point_left, features_3d_msg->features_left.key_points[i]);
        std::copy(&(l_features_msg->descriptor_data[index_left * descriptor_length]), 
                  &(l_features_msg->descriptor_data[(index_left + 1) * descriptor_length]),
                  &(features_3d_msg->features_left.descriptor_data[i * descriptor_length]));
      }
      */

      pub_point_cloud_.publish(point_cloud);
      pub_features_3d_.publish(features_3d_msg);

      ros::WallTime end_time = ros::WallTime::now();
      NODELET_INFO("%zu left, %zu right features, %zu matches, %f sec.", 
          key_points_left.size(), key_points_right.size(), matches.size(), (end_time - start_time).toSec());
    }

  private:

    message_filters::Subscriber<vision_msgs::Features> sub_l_features_, sub_r_features_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_l_info_, sub_r_info_;
    typedef message_filters::sync_policies::ExactTime<vision_msgs::Features,
        vision_msgs::Features, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactPolicy;
    typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
    boost::shared_ptr<ExactSync> exact_sync_;
    bool subscribed_; // stores if anyone is subscribed

    // Publications
    ros::Publisher pub_point_cloud_;
    ros::Publisher pub_features_3d_;

    // the stereo matching constraints
    double max_y_diff_, max_angle_diff_;
    int max_size_diff_;

    // threshold for matching
    double matching_threshold_;
};

} // end of namespace


#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(feature_extraction, 
  StereoFeatureMatcher, 
  feature_extraction_ros::StereoFeatureMatcherNodelet, nodelet::Nodelet);

