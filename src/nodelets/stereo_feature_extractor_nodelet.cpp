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

#include <vision_msgs/StereoFeatures.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include "stereo_camera_model.h"
#include "stereo_feature_extractor.h"
#include "feature_extractor_factory.h"
#include "drawing.h"

#include "stereo_feature_extraction/ExtractFeatures.h" // generated srv header

namespace stereo_feature_extraction
{

class StereoFeatureExtractorNodelet : public nodelet::Nodelet
{
  public:
    StereoFeatureExtractorNodelet() :
        stereo_camera_model_(new StereoCameraModel())
    { }


  private:
    virtual void onInit()
    {
        ros::NodeHandle nh = getNodeHandle();
        ros::NodeHandle& private_nh = getPrivateNodeHandle();
        it_.reset(new image_transport::ImageTransport(nh));
        subscribed_ = false; // no subscription yet
        
        ros::SubscriberStatusCallback connect_cb = 
            boost::bind(&StereoFeatureExtractorNodelet::connectCb, this);

        pub_features_  = nh.advertise<vision_msgs::StereoFeatures>("stereo_features",  1, 
                connect_cb, connect_cb);

        pub_points_ = nh.advertise<sensor_msgs::PointCloud2>("stereo_feature_points", 1,
                connect_cb, connect_cb);

        pub_debug_image_ = nh.advertise<sensor_msgs::Image>("stereo_features_debug_image", 1,
                connect_cb, connect_cb);

        // Synchronize inputs. Topic subscriptions happen on demand in the 
        // connection callback. Optionally do approximate synchronization.
        int queue_size;
        private_nh.param("queue_size", queue_size, 5);
        bool approx;
        private_nh.param("approximate_sync", approx, false);

        private_nh.param("max_y_diff", max_y_diff_, 2.0);
        private_nh.param("max_angle_diff", max_angle_diff_, 2.0);
        private_nh.param("max_size_diff", max_size_diff_, 2);

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
            stereo_feature_extractor_.setFeatureExtractor(feature_extractor);
        }
        stereo_feature_extractor_.setCameraModel(stereo_camera_model_);

        NODELET_INFO_STREAM("Parameters: max_y_diff = " << max_y_diff_
                  << " max_angle_diff = " << max_angle_diff_
                  << " max_size_diff = " << max_size_diff_
                  << " feature_extractor = " << feature_extractor_name);
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

        // advertise the service
        service_server_ = nh.advertiseService("extract_features", 
                &StereoFeatureExtractorNodelet::extractFeaturesSrvCb, this);


        NODELET_INFO("Waiting for client subscriptions.");
    }

    // Handles (un)subscribing when clients (un)subscribe
    void connectCb()
    {
        if (pub_features_.getNumSubscribers() == 0 && 
            pub_debug_image_.getNumSubscribers() == 0 &&
            pub_points_.getNumSubscribers() == 0)
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
        vision_msgs::StereoFeaturesPtr features_msg =
            extractFeatures(*l_image_msg, *r_image_msg,
                            *l_info_msg, *r_info_msg);

        if (features_msg.get() != NULL)
        {
            pub_features_.publish(features_msg);

            if (pub_points_.getNumSubscribers() > 0)
            {
                sensor_msgs::PointCloud2Ptr point_cloud =
                    boost::make_shared<sensor_msgs::PointCloud2>();
                *point_cloud = features_msg->world_points;
                pub_points_.publish(point_cloud);
            }
        }
    }

    vision_msgs::StereoFeaturesPtr extractFeatures(
            const sensor_msgs::Image& l_image_msg,
            const sensor_msgs::Image& r_image_msg,
            const sensor_msgs::CameraInfo& l_info_msg,
            const sensor_msgs::CameraInfo& r_info_msg)
    {
        try
        {
            // bridge to opencv
            namespace enc = sensor_msgs::image_encodings;
            cv_bridge::CvImageConstPtr cv_ptr_left;
            cv_bridge::CvImageConstPtr cv_ptr_right;
            cv_ptr_left = cv_bridge::toCvCopy(l_image_msg, enc::BGR8);
            cv_ptr_right = cv_bridge::toCvCopy(r_image_msg, enc::BGR8);
            
            // Update the camera model
            stereo_camera_model_->fromCameraInfo(l_info_msg, r_info_msg);

            const cv::Mat& left_image = cv_ptr_left->image;
            const cv::Mat& right_image = cv_ptr_right->image;

            // Calculate stereo features
            std::vector<StereoFeature> stereo_features = 
                stereo_feature_extractor_.extract(left_image, right_image, 
                        max_y_diff_, max_angle_diff_, max_size_diff_);

            NODELET_INFO("%zu stereo features extracted.", stereo_features.size());
            if (stereo_features.size() == 0)
            {
                return vision_msgs::StereoFeaturesPtr();
            }

            // Fill in new features message
            vision_msgs::StereoFeaturesPtr features_msg = 
                boost::make_shared<vision_msgs::StereoFeatures>();
            features_msg->header = l_image_msg.header;

            // 3D points
            sensor_msgs::PointCloud2& points_msg = features_msg->world_points;
            points_msg.header = l_image_msg.header;

            points_msg.height = stereo_features.size();
            points_msg.width  = 1;
            points_msg.fields.resize(4);
            points_msg.fields[0].name = "x";
            points_msg.fields[0].offset = 0;
            points_msg.fields[0].count = 1;
            points_msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
            points_msg.fields[1].name = "y";
            points_msg.fields[1].offset = 4;
            points_msg.fields[1].count = 1;
            points_msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
            points_msg.fields[2].name = "z";
            points_msg.fields[2].offset = 8;
            points_msg.fields[2].count = 1;
            points_msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
            points_msg.fields[3].name = "rgb";
            points_msg.fields[3].offset = 12;
            points_msg.fields[3].count = 1;
            points_msg.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
            points_msg.point_step = 16;
            points_msg.row_step = points_msg.point_step * points_msg.width;
            points_msg.data.resize(points_msg.row_step * points_msg.height);
            points_msg.is_dense = false; // there may be invalid points

            features_msg->features.resize(stereo_features.size());
            for (size_t i = 0; i < stereo_features.size(); ++i)
            {
                // fill data of
                // point cloud message
                int offset = i * points_msg.point_step;
                const cv::Point3d& point = stereo_features[i].world_point;
                float x = point.x;
                float y = point.y;
                float z = point.z;
                // pack data into point message data
                memcpy(&points_msg.data[offset + 0], &x, sizeof(float));
                memcpy(&points_msg.data[offset + 4], &y, sizeof(float));
                memcpy(&points_msg.data[offset + 8], &z, sizeof(float));
                cv::Vec3b bgr = stereo_features[i].color;
                int32_t rgb_packed = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
                memcpy(&points_msg.data[offset + 12], &rgb_packed, sizeof(int32_t));

                // feature array
                features_msg->features[i].x = stereo_features[i].key_point_left.pt.x;
                features_msg->features[i].y = stereo_features[i].key_point_left.pt.y;
                features_msg->features[i].descriptor = stereo_features[i].descriptor;
            }

            if (pub_debug_image_.getNumSubscribers() > 0)
            {
                cv::Mat canvas;
                drawStereoFeatures(canvas, cv_ptr_left->image, 
                                    cv_ptr_right->image, stereo_features);
                cv_bridge::CvImage cv_image;
                cv_image.header = cv_ptr_left->header;
                cv_image.encoding = cv_ptr_left->encoding;
                cv_image.image = canvas;
                pub_debug_image_.publish(cv_image.toImageMsg());
            }

            return features_msg;
        }
        catch (cv_bridge::Exception& e)
        {
            NODELET_ERROR("cv_bridge exception: %s", e.what());
        }
        // if anything went wrong, return empty msg
        return vision_msgs::StereoFeaturesPtr();
    }

    /**
    * implementation of the service
    */
    bool extractFeaturesSrvCb(ExtractFeatures::Request& request,
                         ExtractFeatures::Response& response)
    {
         vision_msgs::StereoFeaturesPtr features = extractFeatures(
            request.left_image, request.right_image,
            request.left_camera_info, request.right_camera_info);
         if (features.get() == NULL)
         {
             return false;
         }
         else
         {
            response.features = *features;
            return true;
         }
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

    // Publications
    ros::Publisher pub_features_;
    ros::Publisher pub_debug_image_;
    ros::Publisher pub_points_;

    // Processing state (note: only safe because we're single-threaded!)
    image_geometry::StereoCameraModel model_;
    cv::Mat_<cv::Vec3f> points_mat_; // scratch buffer

    // Error reporting when input topics are not advertised
    boost::shared_ptr<image_proc::AdvertisementChecker> check_inputs_;

    // the camera model
    StereoCameraModel::Ptr stereo_camera_model_;

    // the extractor
    StereoFeatureExtractor stereo_feature_extractor_;

    // the srv
    ros::ServiceServer service_server_;

    // matching parameters
    double max_y_diff_;
    double max_angle_diff_;
    int max_size_diff_;
 
};

PLUGINLIB_DECLARE_CLASS(stereo_feature_extraction, 
    stereo_feature_extractor, 
    stereo_feature_extraction::StereoFeatureExtractorNodelet, nodelet::Nodelet);
}

