#include <opencv2/features2d/features2d.hpp>

void checkEqual(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, double tolerance=1e-6)
{
  EXPECT_NEAR( kp1.pt.x,     kp2.pt.x, tolerance);
  EXPECT_NEAR( kp1.pt.y,     kp2.pt.y, tolerance);
  EXPECT_NEAR( kp1.size,     kp2.size, tolerance);
  EXPECT_NEAR( kp1.angle,    kp2.angle, tolerance);
  EXPECT_NEAR( kp1.response, kp2.response, tolerance);
  EXPECT_EQ(   kp1.octave,   kp2.octave);
  EXPECT_EQ(   kp1.class_id, kp2.class_id);
}

void checkEqual(const cv::Mat mat1, const cv::Mat mat2, double tolerance=1e-6)
{
  EXPECT_EQ(mat1.rows, mat2.rows);
  EXPECT_EQ(mat1.cols, mat2.cols);
  EXPECT_EQ(mat1.type(), mat2.type());

  unsigned char* d1 = mat1.data;
  unsigned char* d2 = mat2.data;

  for (size_t i = 0; i < mat1.rows * mat1.cols * mat1.elemSize(); ++i)
  {
    EXPECT_NEAR(*d1, *d2, tolerance);
    ++d1;
    ++d2;
  }
}

void checkEqual(const cv::Point3d& p1, const cv::Point3d& p2, double tolerance=1e-6)
{
  EXPECT_NEAR(p1.x, p2.x, tolerance);
  EXPECT_NEAR(p1.y, p2.y, tolerance);
  EXPECT_NEAR(p1.z, p2.z, tolerance);
}

template<class T>
void checkEqual(const std::vector<T>& t1, const std::vector<T>& t2, double tolerance=1e-6)
{
  ASSERT_EQ(t1.size(), t2.size());
  for (size_t i = 0; i < t1.size(); ++i)
    checkEqual(t1[i], t2[i]);
}

