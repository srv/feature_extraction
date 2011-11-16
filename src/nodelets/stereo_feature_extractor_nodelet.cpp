#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <boost/timer.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <image_geometry/stereo_camera_model.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/Features3D.h>

#include "feature_matching/stereo_feature_matcher.h"
#include "feature_extraction/feature_extractor_factory.h"
#include "feature_extraction_ros/conversions.h"

#include <opencv2/highgui/highgui.hpp>

namespace feature_extraction_ros
{
class StereoFeatureExtractorNodelet : public nodelet::Nodelet
{
  public:

    StereoFeatureExtractorNodelet() { }

    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

    virtual void onInit()
    {
        ros::NodeHandle nh = getNodeHandle();
        ros::NodeHandle& nh_private = getPrivateNodeHandle();
        subscribed_ = false; // no subscription yet
        
        ros::SubscriberStatusCallback connect_cb = 
            boost::bind(&StereoFeatureExtractorNodelet::connectCb, this);

        pub_point_cloud_ = nh_private.advertise<PointCloud>("point_cloud", 1,
                connect_cb, connect_cb);

        pub_features_3d_ = nh_private.advertise<vision_msgs::Features3D>("features_3d", 1,
                connect_cb, connect_cb);

        int queue_size;
        nh_private.param("queue_size", queue_size, 5);

        nh_private.param("max_y_diff", max_y_diff_, 2.0);
        nh_private.param("max_angle_diff", max_angle_diff_, 2.0);
        nh_private.param("max_size_diff", max_size_diff_, 2);
        nh_private.param("min_depth", min_depth_, 0.2);
        nh_private.param("max_depth", max_depth_, 5.0);
        nh_private.param("matching_threshold", matching_threshold_, 0.8);

        std::string feature_extractor_name;
        private_nh.param("feature_extractor", feature_extractor_name, 
                std::string("SmartSURF"));
        feature_extraction::FeatureExtractor::Ptr feature_extractor = 
            feature_extraction::FeatureExtractorFactory::create(feature_extractor_name);
        if (feature_extractor.get() == NULL)
        {
            NODELET_FATAL("Cannot create feature extractor with name %s",
                    feature_extractor_name.c_str());
        }
        else
        {
            feature_extractor->setMaxNumKeyPoints(max_num_key_points);
            stereo_feature_extractor_.setFeatureExtractor(feature_extractor);
        }

        NODELET_INFO_STREAM("Parameters: \n"
                  " max_y_diff = " << max_y_diff_  << "\n"
                  " max_angle_diff = " << max_angle_diff_ << "\n"
                  " max_size_diff = " << max_size_diff_ << "\n"
                  " depth min/max = " << min_depth_ << "/" << max_depth_ << "\n"
                  " matching_threshold = " << matching_threshold_);

        // Synchronize inputs. Topic subscriptions happen on demand in the 
        // connection callback.
        exact_sync_.reset(new ExactSync(ExactPolicy(queue_size), sub_l_image_, sub_r_image_, sub_l_info_, sub_r_info_));
        exact_sync_->registerCallback(boost::bind(&StereoFeatureExtractorNodelet::imageCb, this, _1, _2, _3, _4));

        NODELET_INFO("Waiting for client subscriptions.");
    }

    // Handles (un)subscribing when clients (un)subscribe
    void connectCb()
    {
        if (pub_features_3d_.getNumSubscribers() == 0 &&
            pub_point_cloud_.getNumSubscribers() == 0)
        {
            NODELET_INFO("No more clients connected, unsubscribing from inputs.");
            sub_l_image_  .unsubscribe();
            sub_r_image_  .unsubscribe();
            sub_l_info_   .unsubscribe();
            sub_r_info_   .unsubscribe();
            subscribed_ = false;
        }
        else if (!subscribed_)
        {
            NODELET_INFO("Client connected, subscribing to inputs.");
            ros::NodeHandle nh = getNodeHandle();
            // Queue size 1 should be OK; the one that matters is the synchronizer queue size.
            sub_l_image_.subscribe(nh, "image_left", 1);
            sub_r_image_.subscribe(nh, "image_right", 1);
            sub_l_info_ .subscribe(nh, "camera_info_left", 1);
            sub_r_info_ .subscribe(nh, "camera_info_right", 1);
            subscribed_ = true;
        }
    }

    void imageCb(const sensor_msgs::ImageConstPtr& l_image_msg,
                 const sensor_msgs::ImageConstPtr& r_image_msg,
                 const sensor_msgs::CameraInfoConstPtr& l_info_msg,
                 const sensor_msgs::CameraInfoConstPtr& r_info_msg)
    {
      ros::WallTime start_time = ros::WallTime::now();

      std::vector<cv::KeyPoint> key_points_left;
      cv::Mat descriptors_left;
      feature_extraction_ros::fromMsg(*l_features_msg, key_points_left, descriptors_left);

      std::vector<cv::KeyPoint> key_points_right;
      cv::Mat descriptors_right;
      feature_extraction_ros::fromMsg(*r_features_msg, key_points_right, descriptors_right);

      // Update the camera model
      stereo_camera_model_.fromCameraInfo(*l_info_msg, *r_info_msg);
      double min_disparity = stereo_camera_model_.getDisparity(max_depth_);
      double max_disparity = stereo_camera_model_.getDisparity(min_depth_);

      feature_matching::StereoFeatureMatcher::Params params;
      params.max_y_diff = max_y_diff_;
      params.max_angle_diff = max_angle_diff_;
      params.min_disparity = min_disparity;
      params.max_disparity = max_disparity;
      params.max_size_diff = max_size_diff_;

      feature_matching::StereoFeatureMatcher matcher;
      matcher.setParams(params);

      boost::timer timer;

      std::vector<cv::DMatch> matches;
      matcher.match(key_points_left, descriptors_left, key_points_right, descriptors_right, matching_threshold_, matches);
      double matching_time = timer.elapsed();

      timer.restart();

      PointCloud::Ptr point_cloud(new PointCloud());
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
      pub_point_cloud_.publish(point_cloud);
      pub_features_3d_.publish(features_3d_msg);

      double msg_construction_time = timer.elapsed();

      ros::WallTime end_time = ros::WallTime::now();
      NODELET_INFO("%zu left, %zu right features, %zu matches.", 
          key_points_left.size(), key_points_right.size(), matches.size());
      NODELET_INFO("Timings: matching: %f, msg construction: %f, total wall time: %f", 
          matching_time, msg_construction_time, (end_time - start_time).toSec());
    }

  private:

    message_filters::Subscriber<sensor_msgs::Image> sub_l_image, sub_r_image_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_l_info_, sub_r_info_;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image,
        sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactPolicy;
    typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
    boost::shared_ptr<ExactSync> exact_sync_;
    bool subscribed_; // stores if anyone is subscribed

    // Publications
    ros::Publisher pub_point_cloud_;
    ros::Publisher pub_features_3d_;

    // the camera model
    image_geometry::StereoCameraModel stereo_camera_model_;

    // the stereo matching constraints
    double max_y_diff_, max_angle_diff_, min_depth_, max_depth_;
    int max_size_diff_;

    // threshold for matching
    double matching_threshold_;
};

} // end of namespace


#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(feature_extraction, 
    StereoFeatureExtractor, 
    feature_extraction_ros::StereoFeatureExtractorNodelet, nodelet::Nodelet);

