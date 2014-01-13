
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/eigen.h>

#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/common/distances.h>

#include <opencv2/core/core.hpp>

#include "feature_matching/estimate_rigid_transformation.h"

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloud;

PointCloud toPcl_(const std::vector<cv::Point3d>& points)
{
  PointCloud cloud;
  for (size_t i = 0; i < points.size(); ++i)
  {
    pcl::PointXYZ point;
    point.x = points[i].x;
    point.y = points[i].y;
    point.z = points[i].z;
    cloud.points.push_back(point);
  }
  return cloud;
}

void drawSamples_(const PointCloud& point_cloud, int nr_samples, 
    double min_sample_distance, std::vector<int>& sample_indices)
{
  int iterations_without_a_sample = 0;
  int max_iterations_without_a_sample = 3 * point_cloud.points.size();
  sample_indices.clear();
  while ((int)sample_indices.size() < nr_samples)
  {
    // Choose a sample at random
    int sample_index = rand() % point_cloud.points.size();

    // Check to see if the sample is 1) unique and 2) far away from the other samples
    bool valid_sample = true;
    for (size_t i = 0; i < sample_indices.size(); ++i)
    {
      float distance_between_samples = pcl::euclideanDistance(point_cloud.points[sample_index], point_cloud.points[sample_indices[i]]);
      if (sample_index == sample_indices[i] || distance_between_samples < min_sample_distance)
      {
        valid_sample = false;
        break;
      }
    }

    // If the sample is valid, add it to the output
    if (valid_sample)
    {
      sample_indices.push_back(sample_index);
      iterations_without_a_sample = 0;
    }
    else
    {
      ++iterations_without_a_sample;
    }

    // If no valid samples can be found, relax the inter-sample distance requirements
    if (iterations_without_a_sample >= max_iterations_without_a_sample)
    {
      std::cerr << "No valid sample found after " << iterations_without_a_sample 
        << " iterations. Relaxing min sample distance to " << 0.5 * min_sample_distance;
      min_sample_distance *= 0.5;
      iterations_without_a_sample = 0;
    }
  }
}

bool feature_matching::estimateRigidTransformation(
    const std::vector<cv::Point3d> source_points, 
    const std::vector<cv::Point3d> target_points,
    cv::Mat& transform, std::vector<unsigned char>& inliers,
    double inlier_threshold, double min_sample_distance, int max_iterations)
{
  PointCloud source_cloud = toPcl_(source_points);
  PointCloud target_cloud = toPcl_(target_points);
  assert(source_cloud.points.size() == target_cloud.points.size());

  std::vector<int> sample_indices;

  Eigen::Matrix4f final_transformation = Eigen::Matrix4f::Identity();
  std::cout << "min sample distance = " << min_sample_distance << std::endl;

  float lowest_error = 0.0;
  Eigen::Matrix4f transformation;
  PointCloud source_cloud_transformed;
  pcl::registration::TransformationEstimationSVD<PointType, PointType> transformation_estimator;
  for (int i_iter = 0; i_iter < max_iterations; ++i_iter)
  {
    int nr_samples = 3;
    drawSamples_(source_cloud, nr_samples, min_sample_distance, sample_indices);
    transformation_estimator.estimateRigidTransformation(source_cloud, sample_indices, target_cloud, sample_indices, transformation);
    transformPointCloud(source_cloud, source_cloud_transformed, transformation);

    // compute error
    float error = 0.0;
    unsigned int num_inliers = 0;
    for (size_t i = 0; i < source_cloud.points.size(); ++i)
    {
      const pcl::PointXYZ& transformed_point = source_cloud_transformed.points[i];
      const pcl::PointXYZ& corresponding_point = target_cloud.points[i];
      float distance = pcl::euclideanDistance(transformed_point, corresponding_point);
      if (distance < inlier_threshold)
      {
        error += distance / inlier_threshold;
        num_inliers++;
      }
      else
      {
        error += 1.0;
      }
    }

    // If the new error is lower, update the final transformation
    if ((i_iter == 0 || error < lowest_error))
    {
      lowest_error = error;
      final_transformation = transformation;
    }
    /*
    if (num_inliers >= min_num_inliers)
    {
        break;
    }
    */
  }

  // final step: compute using all inliers
  inliers.clear();
  std::vector<int> inlier_indices;
  for (size_t i = 0; i < source_cloud_transformed.points.size(); ++i)
  {
    const pcl::PointXYZ& transformed_point = source_cloud_transformed.points[i];
    const pcl::PointXYZ& corresponding_point = target_cloud.points[i];
    float distance = pcl::euclideanDistance(transformed_point, corresponding_point);
    if (distance < inlier_threshold)
    {
      inlier_indices.push_back(i);
      inliers.push_back(1);
    }
    else
    {
      inliers.push_back(0);
    }
  }
  transformation_estimator.estimateRigidTransformation(source_cloud, inlier_indices, target_cloud, inlier_indices, final_transformation);

  transform.create(3, 4, CV_64F);
  for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 4; ++x)
          transform.at<double>(y, x) = final_transformation(y, x);

  return true;
}
