#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/stereo_camera_model.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "feature_matching/stereo_feature_matcher.h"
#include "feature_matching/stereo_depth_estimator.h"
#include "feature_extraction/key_point_detector_factory.h"
#include "feature_extraction/descriptor_extractor_factory.h"
#include "feature_extraction/features_io.h"

using namespace feature_extraction;
namespace po = boost::program_options;

int main(int argc, char** argv)
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("ileft,L", po::value<std::string>()->required(), "left image file")
    ("iright,R", po::value<std::string>()->required(), "right image file")
    ("cleft,J", po::value<std::string>()->required(), "calibration file of left camera")
    ("cright,K", po::value<std::string>()->required(), "calibration file of right camera")
    ("max_y_diff,Y", po::value<double>()->default_value(2.0), "maximum y difference for matching keypoints")
    ("max_angle_diff,A", po::value<double>()->default_value(4.0), "maximum angle difference for matching keypoints")
    ("max_size_diff,S", po::value<int>()->default_value(5), "maximum size difference for matching keypoints")
    ("key_point_detector,D", po::value<std::string>()->default_value("SmartSURF"), "key point detector")
    ("descriptor_extractor,E", po::value<std::string>()->default_value("SmartSURF"), "descriptor extractor")
    ("matching_threshold,T", po::value<double>()->default_value(0.8), "matching threshold")
    ("cloud_file,C", po::value<std::string>(), "file name for output feature PCD point cloud")
    ("output_features_file,O", po::value<std::string>(), "file name for output (key points, descriptors, 3D points)")
    ("display", "display matching output (blocks while window is open)")
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
  std::string left_image_file = vm["ileft"].as<std::string>();
  std::string right_image_file = vm["iright"].as<std::string>();
  std::string left_calibration_file = vm["cleft"].as<std::string>();
  std::string right_calibration_file = vm["cright"].as<std::string>();
  double max_y_diff = vm["max_y_diff"].as<double>();
  double max_angle_diff = vm["max_angle_diff"].as<double>();
  int max_size_diff = vm["max_size_diff"].as<int>();
  std::string key_point_detector_name = vm["key_point_detector"].as<std::string>();
  std::string descriptor_extractor_name = vm["descriptor_extractor"].as<std::string>();
  double matching_threshold = vm["matching_threshold"].as<double>();

  // create instances
  KeyPointDetector::Ptr key_point_detector =
    KeyPointDetectorFactory::create(key_point_detector_name);
  if (key_point_detector.get() == 0)
  {
    std::cerr << "Cannot create key point detector with name '"
              << key_point_detector_name << "'" << std::endl;
    return EXIT_FAILURE;
  }
  DescriptorExtractor::Ptr descriptor_extractor =
    DescriptorExtractorFactory::create(descriptor_extractor_name);
  if (descriptor_extractor.get() == 0)
  {
    std::cerr << "Cannot create descriptor extractor with name '"
              << descriptor_extractor_name << "'" << std::endl;
    return EXIT_FAILURE;
  }

  // load images (as 1 channel)
  cv::Mat image_left = cv::imread(left_image_file, 0);
  cv::Mat image_right = cv::imread(right_image_file, 0);

  // extract key points and descriptors
  std::vector<cv::KeyPoint> key_points_left;
  key_point_detector->detect(image_left, key_points_left);
  cv::Mat descriptors_left;
  descriptor_extractor->extract(image_left, key_points_left, descriptors_left);

  std::vector<cv::KeyPoint> key_points_right;
  key_point_detector->detect(image_right, key_points_right);
  cv::Mat descriptors_right;
  descriptor_extractor->extract(image_right, key_points_right, descriptors_right);

  std::cout << "Extracted " << key_points_left.size() << " descriptors from left image" << std::endl;
  std::cout << "Extracted " << key_points_right.size() << " descriptors from right image" << std::endl;

  // configure and perform matching
  feature_matching::StereoFeatureMatcher::Params params;
  params.max_y_diff = max_y_diff;
  params.max_angle_diff = max_angle_diff;
  params.max_size_diff = max_size_diff;

  feature_matching::StereoFeatureMatcher matcher;
  matcher.setParams(params);
  std::vector<cv::DMatch> matches;
  matcher.match(key_points_left, descriptors_left, key_points_right,
                descriptors_right, matching_threshold, matches);

  std::cout << "Found " << matches.size() << " matches." << std::endl;

  // calculate 3D world points
  typedef pcl::PointXYZI PointType;
  typedef pcl::PointCloud<PointType> PointCloud;
  PointCloud::Ptr point_cloud(new PointCloud());
  feature_matching::StereoDepthEstimator depth_estimator;
  depth_estimator.loadCameraInfo(left_calibration_file, right_calibration_file);
  std::vector<cv::KeyPoint> matched_key_points;
  cv::Mat matched_descriptors;
  std::vector<cv::Point3d> matched_3d_points;
  for (size_t i = 0; i < matches.size(); ++i)
  {
    int index_left = matches[i].queryIdx;
    int index_right = matches[i].trainIdx;
    cv::Point3d world_point;
    depth_estimator.calculate3DPoint(key_points_left[index_left].pt,
                                     key_points_right[index_right].pt,
                                     world_point);
    matched_key_points.push_back(key_points_left[index_left]);
    matched_3d_points.push_back(world_point);
    matched_descriptors.push_back(descriptors_left.row(index_left));
    PointType point;
    point.x = world_point.x;
    point.y = world_point.y;
    point.z = world_point.z;
    point.intensity = image_left.at<unsigned char>(key_points_left[index_left].pt.y,
        key_points_left[index_left].pt.x) / 255.0;
    point_cloud->push_back(point);
  }

  if (vm.count("display"))
  {
    cv::Mat canvas;
    cv::drawMatches(image_left, key_points_left, 
                    image_right, key_points_right, matches, canvas);
    cv::namedWindow("Matches", 0);
    cv::imshow("Matches", canvas);
    cv::waitKey(0);
  }

  if (vm.count("cloud_file"))
  {
    // save pcl
    std::string file_name = vm["cloud_file"].as<std::string>();
    pcl::io::savePCDFile(file_name, *point_cloud);
  }

  if (vm.count("output_features_file"))
  {
    std::string filename = vm["output_features_file"].as<std::string>();
    feature_extraction::features_io::saveStereoFeatures(
        filename, matched_key_points, matched_descriptors, matched_3d_points);
  }
  return EXIT_SUCCESS;
}

