#include <boost/program_options.hpp>
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>
#include "feature_matching/matching_methods.h"
#include "feature_extraction/features_io.h"
#include "feature_matching/estimate_rigid_transformation.h"

namespace po = boost::program_options;

int main(int argc, char** argv)
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("source_features,S", po::value<std::string>()->required(), "source features file")
    ("target_features,T", po::value<std::string>()->required(), "target features file")
    ("matching_threshold,M", po::value<double>()->default_value(0.8), "ratio threshold for matching")
    ("inlier_threshold,I", po::value<double>()->default_value(0.05), "RANSAC inlier/outlier threshold (distance to matched point in [m])")
    ("min_sample_distance,D", po::value<double>()->default_value(0.20), "Minimum distance between sampled points for RANSAC")
    ("output_file,O", po::value<std::string>(), "filename for output")
  ;

  po::variables_map vm;
  try
  {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error& error)
  {
    std::cerr << "Error parsing program options: " << std::endl;
    std::cerr << "  " << error.what() << std::endl;
    std::cerr << desc << std::endl;
    return EXIT_FAILURE;
  }

  // extract params
  std::string source_file = vm["source_features"].as<std::string>();
  std::string target_file = vm["target_features"].as<std::string>();
  double matching_threshold = vm["matching_threshold"].as<double>();
  double inlier_threshold = vm["inlier_threshold"].as<double>();
  double min_sample_distance = vm["min_sample_distance"].as<double>();

  std::vector<cv::KeyPoint> source_key_points, source_key_points_right, 
    target_key_points, target_key_points_right;
  cv::Mat source_descriptors, target_descriptors;
  std::vector<cv::Point3d> source_world_points, target_world_points;

  feature_extraction::features_io::loadStereoFeatures(
      source_file, source_key_points, source_key_points_right, source_descriptors, source_world_points);
  feature_extraction::features_io::loadStereoFeatures(
      target_file, target_key_points, target_key_points_right, target_descriptors, target_world_points);

  std::vector<cv::DMatch> matches, matches12, matches21;
  feature_matching::matching_methods::thresholdMatching(
      source_descriptors, target_descriptors, matching_threshold, 
      cv::Mat(), matches12);
  feature_matching::matching_methods::thresholdMatching(
      target_descriptors, source_descriptors, matching_threshold, 
      cv::Mat(), matches21);
  feature_matching::matching_methods::crossCheckFilter(
      matches12, matches21, matches);
  std::cout << "Found " << matches.size() << " matches. "
    << "(1->2: " << matches12.size()
    << ", 2->1: " << matches21.size() << ")" << std::endl;

  if (matches.size() < 5)
  {
    std::cerr << "Too few matches to calculate transformation." << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<cv::Point3d> matched_source_points(matches.size());
  std::vector<cv::Point3d> matched_target_points(matches.size());
  for (size_t i = 0; i < matches.size(); ++i)
  {
    int index_source = matches[i].queryIdx;
    int index_target = matches[i].trainIdx;
    matched_source_points[i] = source_world_points[index_source];
    matched_target_points[i] = target_world_points[index_target];
  }

  cv::Mat transform;
  std::vector<unsigned char> inliers;
  feature_matching::estimateRigidTransformation(
      matched_source_points, matched_target_points, transform, inliers,
      inlier_threshold, min_sample_distance);

  std::cout << "Found transform with " << cv::countNonZero(inliers) 
    << " inliers: " << std::endl << transform << std::endl;

  if (vm.count("output_file"))
  {
    std::string filename = vm["output_file"].as<std::string>();
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "transformation" << transform;
    fs << "matches" << "[:";
    for (size_t i = 0; i < inliers.size(); ++i)
    {
      if (inliers[i] > 0)
      {
        fs << matches[i].queryIdx;
        fs << matches[i].trainIdx;
        fs << matches[i].imgIdx;
        fs << matches[i].distance;
      }
    }
    fs << "]";
  }
  return EXIT_SUCCESS;
}

