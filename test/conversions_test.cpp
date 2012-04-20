#include <gtest/gtest.h>
#include <opencv2/features2d/features2d.hpp>

#include <ros/package.h>

#include <vision_msgs/Features3D.h>

#include "feature_extraction_ros/conversions.h"

// msg - cv
void checkEqual(const vision_msgs::KeyPoint& kp_msg, const cv::KeyPoint& kp)
{
  EXPECT_NEAR( kp_msg.x,        kp.pt.x, 1e-6);
  EXPECT_NEAR( kp_msg.y,        kp.pt.y, 1e-6);
  EXPECT_NEAR( kp_msg.size,     kp.size, 1e-6);
  EXPECT_NEAR( kp_msg.angle,    kp.angle, 1e-6);
  EXPECT_NEAR( kp_msg.response, kp.response, 1e-6);
  EXPECT_EQ(kp_msg.octave,      kp.octave);
}

// msg - msg
void checkEqual(const vision_msgs::KeyPoint& kp_msg_1, const vision_msgs::KeyPoint& kp_msg_2)
{
  EXPECT_NEAR( kp_msg_1.x,        kp_msg_2.x, 1e-6);
  EXPECT_NEAR( kp_msg_1.y,        kp_msg_2.y, 1e-6);
  EXPECT_NEAR( kp_msg_1.size,     kp_msg_2.size, 1e-6);
  EXPECT_NEAR( kp_msg_1.angle,    kp_msg_2.angle, 1e-6);
  EXPECT_NEAR( kp_msg_1.response, kp_msg_2.response, 1e-6);
  EXPECT_EQ(kp_msg_1.octave,      kp_msg_2.octave);
  // laplacian not checked here because cv::KeyPoint does not have it
}

// cv - cv
void checkEqual(const cv::KeyPoint& kp_msg, const cv::KeyPoint& kp)
{
  EXPECT_NEAR( kp.pt.x,     kp.pt.x, 1e-6);
  EXPECT_NEAR( kp.pt.y,     kp.pt.y, 1e-6);
  EXPECT_NEAR( kp.size,     kp.size, 1e-6);
  EXPECT_NEAR( kp.angle,    kp.angle, 1e-6);
  EXPECT_NEAR( kp.response, kp.response, 1e-6);
  EXPECT_EQ(kp.octave,      kp.octave);
}

TEST(Conversions, keyPointTest)
{
  vision_msgs::KeyPoint kp_msg;
  kp_msg.x = 100.0;
  kp_msg.y = 200.0;
  kp_msg.size = 300.0;
  kp_msg.angle = 90.0;
  kp_msg.response = 400.0;
  kp_msg.octave = 3;
  kp_msg.laplacian = 500.0;

  cv::KeyPoint kp;
  feature_extraction_ros::fromMsg(kp_msg, kp);
  checkEqual(kp_msg, kp);

  vision_msgs::KeyPoint kp_msg_copy;
  feature_extraction_ros::toMsg(kp, kp_msg_copy);
  checkEqual(kp_msg, kp_msg_copy);
}

class MatTest : public ::testing::TestWithParam<int>
{
};


TEST_P(MatTest, matTest)
{
  cv::Mat mat(20, 30, GetParam());
  cv::randu(mat, 0, 256);

  vision_msgs::Mat mat_msg;
  feature_extraction_ros::toMsg(mat, mat_msg);

  EXPECT_EQ(mat_msg.rows, 20);
  EXPECT_EQ(mat_msg.cols, 30);
  EXPECT_EQ(mat_msg.type, mat.type());

  ASSERT_EQ(mat_msg.data.size(), 20 * 30 * mat.elemSize());

  unsigned char* mat_data = mat.data;
  for (size_t i = 0; i < mat_msg.data.size(); ++i)
  {
    EXPECT_EQ(mat_msg.data[i], *mat_data);
    mat_data++;
  }

  cv::Mat mat_copy;
  feature_extraction_ros::fromMsg(mat_msg, mat_copy);

  EXPECT_EQ(mat.rows, mat_copy.rows);
  EXPECT_EQ(mat.cols, mat_copy.cols);
  EXPECT_EQ(mat.type(), mat_copy.type());

  unsigned char* d1 = mat.data;
  unsigned char* d2 = mat_copy.data;

  for (size_t i = 0; i < 20 * 30 * mat.elemSize(); ++i)
  {
    EXPECT_EQ(*d1, *d2);
    ++d1;
    ++d2;
  }
  
}

INSTANTIATE_TEST_CASE_P(MatTests, MatTest,
                        ::testing::Values(CV_8UC1, CV_32FC1, CV_64FC1));

TEST(Conversions, featuresTest)
{
  size_t num_key_points = 200;
  size_t descriptor_length = 64;

  std::vector<cv::KeyPoint> key_points(num_key_points);
  for (size_t i = 0; i < num_key_points; ++i)
  {
    key_points[i].pt.x = 10.0 * i;
    key_points[i].pt.y = 20.0 * i;
    key_points[i].size = 30.0 * i;
    key_points[i].angle = 0.2 * i;
    key_points[i].response = 1.5 * i;
    key_points[i].octave = i % 8;
  }

  cv::Mat descriptors(num_key_points, descriptor_length, CV_32F);
  cv::randu(descriptors, cv::Scalar(0), cv::Scalar(1000));

  vision_msgs::Features features_msg;
  feature_extraction_ros::toMsg(key_points, descriptors, features_msg);
  
  ASSERT_EQ(features_msg.key_points.size(), key_points.size());
  for (size_t i = 0; i < features_msg.key_points.size(); ++i)
  {
    checkEqual(features_msg.key_points[i], key_points[i]);
  }

  std::vector<cv::KeyPoint> key_points_copy;
  cv::Mat descriptors_copy;
  feature_extraction_ros::fromMsg(features_msg, key_points_copy, descriptors_copy);

  ASSERT_EQ(key_points.size(), key_points_copy.size());
  for (size_t i = 0; i < key_points.size(); ++i)
  {
    checkEqual(key_points[i], key_points_copy[i]);
  }

  ASSERT_EQ(descriptors.rows, descriptors_copy.rows);
  ASSERT_EQ(descriptors.cols, descriptors_copy.cols);

  cv::MatConstIterator_<float> it1 = descriptors.begin<float>();
  cv::MatConstIterator_<float> it2 = descriptors_copy.begin<float>();

  while (it1 != descriptors.end<float>())
  {
    EXPECT_NEAR(*it1++, *it2++, 1e-6);
  }
}

TEST(Conversions, emptyTest)
{
  std::vector<cv::KeyPoint> key_points;
  cv::Mat descriptors;
  vision_msgs::Features features_msg;
  feature_extraction_ros::toMsg(key_points, descriptors, features_msg);
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

