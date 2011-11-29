#include <ros/package.h>

#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "feature_extraction/key_point_detector_factory.h"

using namespace feature_extraction;

class KeyPointDetectorsTest : public ::testing::TestWithParam<const char*>
{
};

TEST_P(KeyPointDetectorsTest, runTest)
{
  std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
  cv::Mat image = cv::imread(path + "/data/black_box.jpg");
  ASSERT_FALSE(image.empty());
 
  KeyPointDetector::Ptr detector = 
    KeyPointDetectorFactory::create(GetParam());
  std::vector<cv::KeyPoint> key_points;
  double time = (double)cv::getTickCount();
  detector->detect(image, key_points);
  time = ((double)cv::getTickCount() - time)/cv::getTickFrequency() * 1000;
  EXPECT_GT(key_points.size(), 0);
  std::cout << GetParam() << " detected " << key_points.size() 
      << " key points in " << time << "ms." << std::endl;
}

const char* detector_names[] = {
  "SmartSURF",
  "CvFAST",
  "CvSTAR",
  "CvSIFT",
  "CvSURF",
  "CvORB",
  "CvMSER",
  "CvGFTT",
  "CvHARRIS",
  "CvDense",
  "CvSimpleBlob" };

INSTANTIATE_TEST_CASE_P(KeyPointDetectorsTests, KeyPointDetectorsTest,
                        ::testing::ValuesIn(detector_names));

