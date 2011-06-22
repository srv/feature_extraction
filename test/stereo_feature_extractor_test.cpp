#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>

#include "ros/package.h"

#include "stereo_feature_extractor.h"
#include "feature_extractor_factory.h"
#include "sensor_msgs/CameraInfo.h"
#include "image_geometry/stereo_camera_model.h"


using namespace stereo_feature_extraction;

TEST(StereoFeatureExtractor, runTest)
{
    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);

    // load image
    cv::Mat image_left = cv::imread(path + "/data/black_box.jpg");
    ASSERT_FALSE(image_left.empty());
    cv::Mat image_right = cv::Mat(image_left.rows, image_left.cols,
            image_left.type());

    std::vector<double> distances;
    distances.push_back(0.3);
    distances.push_back(0.4);
    distances.push_back(0.5);
    distances.push_back(0.6);
    distances.push_back(0.7);
    distances.push_back(0.8);
    distances.push_back(0.9);
    distances.push_back(1.0);
    distances.push_back(1.2);
    distances.push_back(1.4);
    distances.push_back(1.6);
    distances.push_back(1.8);
    distances.push_back(2.0);
    distances.push_back(2.5);
    distances.push_back(3.0);
    distances.push_back(3.5);
    distances.push_back(4.0);
    distances.push_back(4.5);
    distances.push_back(5.0);

    for (size_t k = 0; k < distances.size(); ++k)
    {
        double shift = 10.0 / distances[k];
        cv::Mat transform = cv::Mat::eye(2, 3, CV_32F);
        transform.at<float>(0, 2) = -shift;
        cv::warpAffine(image_left, image_right, transform, image_left.size());

        FeatureExtractor::Ptr feature_extractor = 
            FeatureExtractorFactory::create("SURF");

        // read calibration data to fill stereo camera model
        std::string left_file = path + "/data/artificial_calibration_left.yaml";
        std::string right_file = path + "/data/artificial_calibration_right.yaml";

        StereoCameraModel::Ptr stereo_camera_model = 
            StereoCameraModel::Ptr(new StereoCameraModel());
        stereo_camera_model->fromCalibrationFiles(left_file, right_file);

        StereoFeatureExtractor extractor;
        extractor.setFeatureExtractor(feature_extractor);
        extractor.setMatchMethod(StereoFeatureExtractor::KEY_POINT_TO_BLOCK);
        extractor.setCameraModel(stereo_camera_model);

        double max_y_diff = 0.5;
        double max_angle_diff = 2.0;
        int max_size_diff = 0;
        std::vector<StereoFeature> stereo_features = 
            extractor.extract(image_left, image_right,
                    max_y_diff, max_angle_diff, max_size_diff);
        std::cout << "Found " << stereo_features.size() << " stereo features." << std::endl;

        ASSERT_TRUE(stereo_features.size() > 0);

        // the artificial calibration has a baseline length of 10, so
        // a disparity of 10 means 1m distance
        double expected_distance = 10.0 / shift;
        double error = 0.0;
        for (size_t i = 0; i < stereo_features.size(); ++i)
        {
            double diff = stereo_features[i].world_point.z - expected_distance;
            error += std::abs(diff); 
        }
        error /= stereo_features.size();
        std::cout << "mean error for distance " << expected_distance << ": " << error << std::endl;
    }
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

