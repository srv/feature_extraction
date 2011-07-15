#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_proc/advertisement_checker.h>

#include <image_geometry/stereo_camera_model.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/RegionOfInterest.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include "stereo_camera_model.h"
#include "stereo_feature_extractor.h"
#include "feature_extractor_factory.h"
#include "drawing.h"

namespace stereo_feature_extraction
{

struct Feature
{
    float data[64];
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
}

POINT_CLOUD_REGISTER_POINT_STRUCT (stereo_feature_extraction::Feature,
                                (float[64], data, data)
                                )

namespace stereo_feature_extraction
{
class StereoFeatureExtractorNodelet : public nodelet::Nodelet
{
  public:
    StereoFeatureExtractorNodelet() :
        stereo_camera_model_(new StereoCameraModel())
    { }

    typedef pcl::PointCloud<Feature> FeatureCloud;
    typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

  private:
    virtual void onInit()
    {
        ros::NodeHandle nh = getNodeHandle();
        ros::NodeHandle& private_nh = getPrivateNodeHandle();
        it_.reset(new image_transport::ImageTransport(nh));
        subscribed_ = false; // no subscription yet
        
        ros::SubscriberStatusCallback connect_cb = 
            boost::bind(&StereoFeatureExtractorNodelet::connectCb, this);

        pub_point_cloud_ = nh.advertise<PointCloud>("point_cloud", 1,
                connect_cb, connect_cb);

        pub_feature_cloud_ = nh.advertise<FeatureCloud>("feature_cloud", 1,
                connect_cb, connect_cb);

        pub_debug_image_ = nh.advertise<sensor_msgs::Image>("stereo_features_debug_image", 1,
                connect_cb, connect_cb);

        sub_region_of_interest_ = nh.subscribe("region_of_interest", 10, &StereoFeatureExtractorNodelet::setRegionOfInterest, this);

        // Synchronize inputs. Topic subscriptions happen on demand in the 
        // connection callback. Optionally do approximate synchronization.
        int queue_size;
        private_nh.param("queue_size", queue_size, 5);
        bool approx;
        private_nh.param("approximate_sync", approx, false);

        double max_y_diff, max_angle_diff, min_depth, max_depth;
        int max_size_diff, max_num_key_points;
        private_nh.param("max_y_diff", max_y_diff, 2.0);
        private_nh.param("max_angle_diff", max_angle_diff, 2.0);
        private_nh.param("max_size_diff", max_size_diff, 2);
        private_nh.param("min_depth", min_depth, 0.2);
        private_nh.param("max_depth", max_depth, 5.0);
        private_nh.param("max_num_key_points", max_num_key_points, 5000);

        std::string feature_extractor_name;
        private_nh.param("feature_extractor", feature_extractor_name, 
                std::string("SURF"));
        FeatureExtractor::Ptr feature_extractor = 
            FeatureExtractorFactory::create(feature_extractor_name);
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
        stereo_feature_extractor_.setCameraModel(stereo_camera_model_);

        stereo_feature_extractor_.setMaxYDiff(max_y_diff);
        stereo_feature_extractor_.setMaxAngleDiff(max_angle_diff);
        stereo_feature_extractor_.setMaxSizeDiff(max_size_diff);
        stereo_feature_extractor_.setMinDepth(min_depth);
        stereo_feature_extractor_.setMaxDepth(max_depth);

        NODELET_INFO_STREAM("Parameters: max_y_diff = " << max_y_diff
                  << " max_angle_diff = " << max_angle_diff
                  << " max_size_diff = " << max_size_diff
                  << " depth min/max = " << min_depth << "/" << max_depth
                  << " feature_extractor = " << feature_extractor_name
                  << " max_num_key_points = " << max_num_key_points);
        if (approx)
        {
            approximate_sync_.reset(
                    new ApproximateSync(ApproximatePolicy(queue_size),
                                        sub_l_image_, sub_r_image_,
                                        sub_l_info_, sub_r_info_));
            approximate_sync_->registerCallback(
                    boost::bind(&StereoFeatureExtractorNodelet::imageCb,
                                this, _1, _2, _3, _4));
        }
        else
        {
            exact_sync_.reset(new ExactSync(ExactPolicy(queue_size),
                                            sub_l_image_, sub_r_image_,
                                            sub_l_info_, sub_r_info_));
            exact_sync_->registerCallback(
                    boost::bind(&StereoFeatureExtractorNodelet::imageCb,
                                this, _1, _2, _3, _4));
        }

        NODELET_INFO("Waiting for client subscriptions.");
    }

    // Handles (un)subscribing when clients (un)subscribe
    void connectCb()
    {
        if (pub_debug_image_.getNumSubscribers() == 0 &&
            pub_feature_cloud_.getNumSubscribers() == 0 &&
            pub_point_cloud_.getNumSubscribers() == 0)
        {
            NODELET_INFO("No more clients connected, unsubscribing from camera.");
            sub_l_image_  .unsubscribe();
            sub_r_image_  .unsubscribe();
            sub_l_info_   .unsubscribe();
            sub_r_info_   .unsubscribe();
            subscribed_ = false;
        }
        else if (!subscribed_)
        {
            NODELET_INFO("Client connected, subscribing to camera.");
            ros::NodeHandle &nh = getNodeHandle();
            // Queue size 1 should be OK; the one that matters is the synchronizer queue size.
            sub_l_image_  .subscribe(*it_, "left/image_rect", 1);
            sub_r_image_  .subscribe(*it_, "right/image_rect", 1);
            sub_l_info_   .subscribe(nh,   "left/camera_info", 1);
            sub_r_info_   .subscribe(nh,   "right/camera_info", 1);
            subscribed_ = true;
        }
    }

    void imageCb(const sensor_msgs::ImageConstPtr& l_image_msg,
                 const sensor_msgs::ImageConstPtr& r_image_msg,
                 const sensor_msgs::CameraInfoConstPtr& l_info_msg,
                 const sensor_msgs::CameraInfoConstPtr& r_info_msg)
    {
        if (region_of_interest_.area() == 0)
        {
            // invalid roi, set to image size
            region_of_interest_ = 
                cv::Rect(0, 0, l_image_msg->width, l_image_msg->height);
        }
           
        try
        {
            // bridge to opencv
            namespace enc = sensor_msgs::image_encodings;
            cv_bridge::CvImageConstPtr cv_ptr_left;
            cv_bridge::CvImageConstPtr cv_ptr_right;
            cv_ptr_left = cv_bridge::toCvCopy(l_image_msg, enc::BGR8);
            cv_ptr_right = cv_bridge::toCvCopy(r_image_msg, enc::BGR8);
            
            // Update the camera model
            stereo_camera_model_->fromCameraInfo(*l_info_msg, *r_info_msg);

            cv::Mat left_image = cv_ptr_left->image;
            cv::Mat right_image = cv_ptr_right->image;

            // Calculate stereo features
            StereoFeatureSet stereo_feature_set = 
                stereo_feature_extractor_.extract(left_image, right_image);
            std::vector<StereoFeature>& stereo_features = 
                stereo_feature_set.stereo_features;

            NODELET_INFO("%zu stereo features extracted.", stereo_features.size());
            if (stereo_features.size() == 0)
            {
                return;
            }

            FeatureCloud::Ptr feature_cloud(new FeatureCloud());
            feature_cloud->header = l_image_msg->header;
            feature_cloud->points.resize(stereo_features.size());
            PointCloud::Ptr point_cloud(new PointCloud());
            point_cloud->header = l_image_msg->header;
            point_cloud->points.resize(stereo_features.size());
            for (size_t i = 0; i < stereo_features.size(); ++i)
            {
                const cv::Point3d& point = stereo_features[i].world_point;
                pcl::PointXYZRGB& pcl_point = point_cloud->points[i];
                pcl_point.x = point.x;
                pcl_point.y = point.y;
                pcl_point.z = point.z;
                cv::Vec3b bgr = stereo_features[i].color;
                int32_t rgb_packed = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
                memcpy(&pcl_point.rgb, &rgb_packed, sizeof(int32_t));

                Feature& feature = feature_cloud->points[i];
                std::vector<float> descriptor = stereo_features[i].descriptor;
                std::copy(descriptor.begin(), descriptor.end(), feature.data);
            }

            if (pub_debug_image_.getNumSubscribers() > 0)
            {
                cv::Mat canvas;
                drawStereoFeatures(canvas, cv_ptr_left->image, 
                                    cv_ptr_right->image, stereo_features);
                cv::rectangle(canvas, region_of_interest_.tl(),
                        region_of_interest_.br(), cv::Scalar(0, 0, 255), 3);
                cv_bridge::CvImage cv_image;
                cv_image.header = cv_ptr_left->header;
                cv_image.encoding = cv_ptr_left->encoding;
                cv_image.image = canvas;
                pub_debug_image_.publish(cv_image.toImageMsg());
            }

            pub_point_cloud_.publish(point_cloud);
            pub_feature_cloud_.publish(feature_cloud);
        }
        catch (cv_bridge::Exception& e)
        {
            NODELET_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    void setRegionOfInterest(const sensor_msgs::RegionOfInterestConstPtr& roi_msg)
    {
        region_of_interest_.x = roi_msg->x_offset;
        region_of_interest_.y = roi_msg->y_offset;
        region_of_interest_.width = roi_msg->width;
        region_of_interest_.height = roi_msg->height;
        ROS_INFO("Region of interest set to: (%i, %i), %ix%i",
                region_of_interest_.x,
                region_of_interest_.y,
                region_of_interest_.width,
                region_of_interest_.height);
        stereo_feature_extractor_.setRegionOfInterest(region_of_interest_);
    }

    boost::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::SubscriberFilter sub_l_image_;
    image_transport::SubscriberFilter sub_r_image_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_l_info_, sub_r_info_;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image,
        sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactPolicy;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
        sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ApproximatePolicy;
    typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
    typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
    boost::shared_ptr<ExactSync> exact_sync_;
    boost::shared_ptr<ApproximateSync> approximate_sync_;
    bool subscribed_; // stores if anyone is subscribed

    ros::Subscriber sub_region_of_interest_;

    // Publications
    ros::Publisher pub_point_cloud_;
    ros::Publisher pub_feature_cloud_;
    ros::Publisher pub_debug_image_;

    // Processing state (note: only safe because we're single-threaded!)
    image_geometry::StereoCameraModel model_;
    cv::Mat_<cv::Vec3f> points_mat_; // scratch buffer

    // the camera model
    StereoCameraModel::Ptr stereo_camera_model_;

    // the extractor
    StereoFeatureExtractor stereo_feature_extractor_;

    // the current roi
    cv::Rect region_of_interest_;
 
};

PLUGINLIB_DECLARE_CLASS(stereo_feature_extraction, 
    stereo_feature_extractor, 
    stereo_feature_extraction::StereoFeatureExtractorNodelet, nodelet::Nodelet);
}

