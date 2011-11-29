#include <ros/package.h>

#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "feature_extraction/key_point_detector_factory.h"
#include "feature_extraction/descriptor_extractor_factory.h"

using namespace feature_extraction;


class DescriptorExtractorsTest : public ::testing::TestWithParam<const char*>
{
};

TEST_P(DescriptorExtractorsTest, runTest)
{
  std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
  cv::Mat image = cv::imread(path + "/data/black_box.jpg");
  ASSERT_FALSE(image.empty());

  ASSERT_GT(image.rows, 200);
  ASSERT_GT(image.cols, 200);

  // some random key points
  cv::RNG rng;
  static const int NUM_KEY_POINTS = 1000;
  std::vector<cv::KeyPoint> key_points(NUM_KEY_POINTS);
  for (size_t i = 0; i < key_points.size(); ++i)
  {
    key_points[i].pt.x = rng.uniform(0.0, static_cast<double>(image.cols - 1));
    key_points[i].pt.y = rng.uniform(0.0, static_cast<double>(image.rows - 1));
    key_points[i].size = rng.uniform(1.0, 100.0);
  }

  DescriptorExtractor::Ptr extractor = 
    DescriptorExtractorFactory::create(GetParam());
  cv::Mat descriptors;
  double time = (double)cv::getTickCount();
  extractor->extract(image, key_points, descriptors);
  time = ((double)cv::getTickCount() - time)/cv::getTickFrequency() * 1000;
  EXPECT_GT(key_points.size(), 0);
  EXPECT_EQ(key_points.size(), descriptors.rows);
  std::cout << GetParam() << " extracted " << descriptors.rows 
      << " descriptors in " << time << "ms." << std::endl;
}

TEST_P(DescriptorExtractorsTest, emptyTest)
{
  std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
  cv::Mat image = cv::imread(path + "/data/black_box.jpg");
  ASSERT_FALSE(image.empty());

  std::vector<cv::KeyPoint> key_points;
  DescriptorExtractor::Ptr extractor = 
    DescriptorExtractorFactory::create(GetParam());
  cv::Mat descriptors;
  extractor->extract(image, key_points, descriptors);
  EXPECT_EQ(key_points.size(), 0);
  EXPECT_EQ(descriptors.rows, 0);
  EXPECT_EQ(descriptors.cols, 0);
}

const char* extractor_names[] = {
  "SmartSURF", 
  "CvSIFT", 
  "CvSURF", 
  "CvORB", 
  "CvBRIEF" };

INSTANTIATE_TEST_CASE_P(DescriptorExtractorTests, DescriptorExtractorsTest,
                        ::testing::ValuesIn(extractor_names));

