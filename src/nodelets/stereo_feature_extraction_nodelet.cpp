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

#include <sensor_msgs/PointCloud2.h>

#include "stereo_camera_model.h"
#include "stereo_feature_extractor.h"
#include "feature_extractor_factory.h"

namespace stereo_feature_extraction
{

class StereoFeatureExtractionNodelet : public nodelet::Nodelet
{
  public:
    StereoFeatureExtractionNodelet() :
        stereo_camera_model_(new StereoCameraModel()),
        stereo_feature_extractor_(FeatureExtractorFactory::create("CvSURF"),
                                  stereo_camera_model_)
    {}

  private:
    virtual void onInit()
    {
        ros::NodeHandle nh = getNodeHandle();
        ros::NodeHandle& private_nh = getPrivateNodeHandle();
        pub = private_nh.advertise<std_msgs::Float64>("out", 10);
        it_.reset(new image_transport::ImageTransport(nh));
        subscribed_ = false; // no subscription yet
        
        ros::SubscriberStatusCallback connect_cb = 
            boost::bind(&StereoFeatureExtractionNodelet::connectCb, this);

        pub_points2_  = nh.advertise<PointCloud2>("stereo_features",  1, 
                connect_cb, connect_cb);

        // Synchronize inputs. Topic subscriptions happen on demand in the 
        // connection callback. Optionally do approximate synchronization.
        int queue_size;
        private_nh.param("queue_size", queue_size, 5);
        bool approx;
        private_nh.param("approximate_sync", approx, false);
        if (approx)
        {
            approximate_sync_.reset(
                    new ApproximateSync(ApproximatePolicy(queue_size),
                                        sub_l_image_, sub_r_image_,
                                        sub_l_info_, sub_r_info_));
            approximate_sync_->registerCallback(
                    boost::bind(&StereoFeatureExtractionNodelet::imageCb,
                                this, _1, _2, _3, _4));
        }
        else
        {
            exact_sync_.reset(new ExactSync(ExactPolicy(queue_size),
                                            sub_l_image_, sub_r_image_,
                                            sub_l_info_, sub_r_info_));
            exact_sync_->registerCallback(
                    boost::bind(&StereoFeatureExtractionNodelet::imageCb,
                                this, _1, _2, _3, _4));
        }

        // Print a warning every 10 seconds until the input topics are advertised
        ros::V_string topics;
        topics.push_back("left/image_rect_color");
        topics.push_back("right/image_rect_color");
        topics.push_back("left/camera_info");
        topics.push_back("right/camera_info");
        check_inputs_.reset(new image_proc::AdvertisementChecker(nh, getName()));
        check_inputs_->start(topics, 60.0);

    }

    // Handles (un)subscribing when clients (un)subscribe
    void connectCb()
    {
        if (pub_points2_.getNumSubscribers() == 0)
        {
            sub_l_image_  .unsubscribe();
            sub_r_image_  .unsubscribe();
            sub_l_info_   .unsubscribe();
            sub_r_info_   .unsubscribe();
            subscribed_ = false;
        }
        else if (!subscribed_)
        {
            ros::NodeHandle &nh = getNodeHandle();
            // Queue size 1 should be OK; the one that matters is the synchronizer queue size.
            sub_l_image_  .subscribe(*it_, "left/image_rect_color", 1);
            sub_r_image_  .subscribe(*it_, "right/image_rect_color", 1);
            sub_l_info_   .subscribe(nh,   "left/camera_info", 1);
            sub_r_info_   .subscribe(nh,   "right/camera_info", 1);
            subscribed_ = true;
        }
    }

    void imageCb(const ImageConstPtr& l_image_msg,
                 const ImageConstPtr& r_image_msg,
                 const CameraInfoConstPtr& l_info_msg,
                 const CameraInfoConstPtr& r_info_msg)
    {
        // Update the camera model
        stereo_camera_model_->fromCameraInfo(l_info_msg, r_info_msg);

        // bridge to opencv
        cv_bridge::CvImagePtr cv_ptr_left;
        cv_bridge::CvImagePtr cv_ptr_right;
        try
        {
            cv_ptr_left = cv_bridge::toCvShare(l_image_msg, enc::BGR8);
            cv_ptr_right = cv_bridge::toCvShare(l_image_msg, enc::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            NODELET_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Calculate stereo features
        double max_y_diff = 5;
        double max_angle_diff = 5;
        double max_size_diff = 2;
        cv::Mat mask;
        std::vector<StereoFeature> stereo_features = stereo_feature_extractor_->extract(
            cv_ptr_left->image, cv_ptr_right->image, mask, mask, 
            max_y_diff, max_angle_diff, max_size_diff);

/*
        // Fill in new PointCloud2 message (2D image-like layout)
        PointCloud2Ptr points_msg = boost::make_shared<PointCloud2>();
        points_msg->header = l_image_msg->header;
        points_msg->height = stereo_features.size();
        points_msg->width  = 1;
        points_msg->fields.resize (4);
        points_msg->fields[0].name = "x";
        points_msg->fields[0].offset = 0;
        points_msg->fields[0].count = 1;
        points_msg->fields[0].datatype = PointField::FLOAT32;
        points_msg->fields[1].name = "y";
        points_msg->fields[1].offset = 4;
        points_msg->fields[1].count = 1;
        points_msg->fields[1].datatype = PointField::FLOAT32;
        points_msg->fields[2].name = "z";
        points_msg->fields[2].offset = 8;
        points_msg->fields[2].count = 1;
        points_msg->fields[2].datatype = PointField::FLOAT32;
        points_msg->fields[3].name = "rgb";
        points_msg->fields[3].offset = 12;
        points_msg->fields[3].count = 1;
        points_msg->fields[3].datatype = PointField::FLOAT32;
        static const int STEP = 16;
        points_msg->point_step = STEP;
        points_msg->row_step = points_msg->point_step * points_msg->width;
        points_msg->data.resize (points_msg->row_step * points_msg->height);
        points_msg->is_dense = false; // there may be invalid points

        float bad_point = std::numeric_limits<float>::quiet_NaN ();
        int offset = 0;
        for (int v = 0; v < mat.rows; ++v)
        {
        for (int u = 0; u < mat.cols; ++u, offset += STEP)
        {
            if (isValidPoint(mat(v,u)))
            {
            // x,y,z,rgba
            memcpy (&points_msg->data[offset + 0], &mat(v,u)[0], sizeof (float));
            memcpy (&points_msg->data[offset + 4], &mat(v,u)[1], sizeof (float));
            memcpy (&points_msg->data[offset + 8], &mat(v,u)[2], sizeof (float));
            }
            else
            {
            memcpy (&points_msg->data[offset + 0], &bad_point, sizeof (float));
            memcpy (&points_msg->data[offset + 4], &bad_point, sizeof (float));
            memcpy (&points_msg->data[offset + 8], &bad_point, sizeof (float));
            }
        }
        }

        // Fill in color
        namespace enc = sensor_msgs::image_encodings;
        const std::string& encoding = l_image_msg->encoding;
        offset = 0;
        if (encoding == enc::MONO8)
        {
        const cv::Mat_<uint8_t> color(l_image_msg->height, l_image_msg->width,
                                        (uint8_t*)&l_image_msg->data[0],
                                        l_image_msg->step);
        for (int v = 0; v < mat.rows; ++v)
        {
            for (int u = 0; u < mat.cols; ++u, offset += STEP)
            {
            if (isValidPoint(mat(v,u)))
            {
                uint8_t g = color(v,u);
                int32_t rgb = (g << 16) | (g << 8) | g;
                memcpy (&points_msg->data[offset + 12], &rgb, sizeof (int32_t));
            }
            else
            {
                memcpy (&points_msg->data[offset + 12], &bad_point, sizeof (float));
            }
            }
        }
        }
        else if (encoding == enc::RGB8)
        {
        const cv::Mat_<cv::Vec3b> color(l_image_msg->height, l_image_msg->width,
                                        (cv::Vec3b*)&l_image_msg->data[0],
                                        l_image_msg->step);
        for (int v = 0; v < mat.rows; ++v)
        {
            for (int u = 0; u < mat.cols; ++u, offset += STEP)
            {
            if (isValidPoint(mat(v,u)))
            {
                const cv::Vec3b& rgb = color(v,u);
                int32_t rgb_packed = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
                memcpy (&points_msg->data[offset + 12], &rgb_packed, sizeof (int32_t));
            }
            else
            {
                memcpy (&points_msg->data[offset + 12], &bad_point, sizeof (float));
            }
            }
        }
        }
        else if (encoding == enc::BGR8)
        {
        const cv::Mat_<cv::Vec3b> color(l_image_msg->height, l_image_msg->width,
                                        (cv::Vec3b*)&l_image_msg->data[0],
                                        l_image_msg->step);
        for (int v = 0; v < mat.rows; ++v)
        {
            for (int u = 0; u < mat.cols; ++u, offset += STEP)
            {
            if (isValidPoint(mat(v,u)))
            {
                const cv::Vec3b& bgr = color(v,u);
                int32_t rgb_packed = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
                memcpy (&points_msg->data[offset + 12], &rgb_packed, sizeof (int32_t));
            }
            else
            {
                memcpy (&points_msg->data[offset + 12], &bad_point, sizeof (float));
            }
            }
        }
        }
        else
        {
        NODELET_WARN_THROTTLE(30, "Could not fill color channel of the point cloud, "
                                "unsupported encoding '%s'", encoding.c_str());
        }

        pub_points2_.publish(points_msg);
*/
        }

    }

    image_transport::SubscriberFilter sub_l_image_;
    image_transport::SubscriberFilter sub_r_image_;
    message_filters::Subscriber<CameraInfo> sub_l_info_, sub_r_info_;
    typedef ExactTime<Image, Image, CameraInfo, CameraInfo> ExactPolicy;
    typedef ApproximateTime<Image, Image, CameraInfo, CameraInfo> ApproximatePolicy;
    typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
    typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
    boost::shared_ptr<ExactSync> exact_sync_;
    boost::shared_ptr<ApproximateSync> approximate_sync_;
    bool subscribed_; // stores if anyone is subscribed

    // Publications
    ros::Publisher pub_points2_;

    // Processing state (note: only safe because we're single-threaded!)
    image_geometry::StereoCameraModel model_;
    cv::Mat_<cv::Vec3f> points_mat_; // scratch buffer

    // Error reporting when input topics are not advertised
    boost::shared_ptr<image_proc::AdvertisementChecker> check_inputs_;

    // the camera model
    StereoCameraModel::Ptr stereo_camera_model_;

    // the extractor
    StereoFeatureExtractor stereo_feature_extractor_;
 
};

PLUGINLIB_DECLARE_CLASS(stereo_feature_extraction, 
    StereoFeatureExtractionNodelet, 
    stereo_feature_extraction::StereoFeatureExtractionNodelet, nodelet::Nodelet);
}

