#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/package.h>

#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "stereo_feature_extractor.h"
#include "feature_extractor_factory.h"
#include "drawing.h"
#include "sensor_msgs/CameraInfo.h"
#include "image_geometry/stereo_camera_model.h"


using namespace stereo_feature_extraction;
namespace po = boost::program_options;

struct TrainTestSet
{
    std::vector<StereoFeature> train_features;
    std::vector<StereoFeature> test_features;
};

bool compareKeyPointSize(const StereoFeature& feature1, const StereoFeature& feature2)
{
    return feature1.key_point_left.size < feature2.key_point_left.size;
}

StereoFeatureExtractor createStandardExtractor(
        const std::string& feature_extractor_name,
        int max_num_key_points)
{
    FeatureExtractor::Ptr feature_extractor = 
        FeatureExtractorFactory::create(feature_extractor_name);
    if (!feature_extractor.get())
    {
        std::cerr << "cannot create feature extractor with name '"
            << feature_extractor_name << "'!" << std::endl;
        return StereoFeatureExtractor();
    }
    feature_extractor->setMaxNumKeyPoints(max_num_key_points);

    // read calibration data to fill stereo camera model
    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
    std::string left_file = path + "/data/real_calibration_left.yaml";
    std::string right_file = path + "/data/real_calibration_right.yaml";

    StereoCameraModel::Ptr stereo_camera_model = 
        StereoCameraModel::Ptr(new StereoCameraModel());
    stereo_camera_model->fromCalibrationFiles(left_file, right_file);

    StereoFeatureExtractor extractor;
    extractor.setFeatureExtractor(feature_extractor);
    extractor.setCameraModel(stereo_camera_model);
    return extractor;
}

void loadImages(int distance, cv::Mat& left_image, cv::Mat& right_image)
{
    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
    std::ostringstream filename;
    filename << path << "/data/left_" << distance << "cm.jpg";
    cv::Mat image_left = cv::imread(filename.str());
    if (image_left.empty())
    {
        throw std::runtime_error("Cannot load image " + filename.str());
    }
    filename.str("");
    filename << path << "/data/right_" << distance << "cm.jpg";
    cv::Mat image_right = cv::imread(filename.str());
    if (image_right.empty())
    {
        throw std::runtime_error("Cannot load image " + filename.str());
    }
    image_left.copyTo(left_image);
    image_right.copyTo(right_image);
    if (image_left.rows != 768 || image_left.cols != 1024 ||
        image_right.rows != 768 || image_right.cols != 1024) 
        throw std::runtime_error("invalid image size, must be 1024x768");
}

pcl::PointCloud<pcl::PointXYZ> createCloud(
        const std::vector<StereoFeature>& stereo_features)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    // Fill in the cloud data
    // we cut off everything that is inside a border of 100px because
    // the test images plane was not big enough
    for (size_t i = 0; i < stereo_features.size (); ++i)
    {
        if (stereo_features[i].key_point_left.pt.x > 100 &&
            stereo_features[i].key_point_left.pt.x < 924 &&
            stereo_features[i].key_point_left.pt.y > 100 &&
            stereo_features[i].key_point_left.pt.y < 668)
        {
            pcl::PointXYZ point;
            point.x = stereo_features[i].world_point.x;
            point.y = stereo_features[i].world_point.y;
            point.z = stereo_features[i].world_point.z;
            cloud.push_back(point);
        }
    }
    return cloud;
}

void runTest(const std::vector<StereoFeature> train_features,
        const std::vector<StereoFeature> test_features)
{
    // the sets must be sorted!
    std::cout << "train set: min size = " 
        << train_features[0].key_point_left.size << " max size = " 
        << train_features[train_features.size() - 1].key_point_left.size
        << std::endl;

    std::cout << "test set: min size = " 
        << test_features[0].key_point_left.size << " max size = " 
        << test_features[test_features.size() - 1].key_point_left.size
        << std::endl;

    pcl::PointCloud<pcl::PointXYZ> train_cloud = createCloud(train_features);
    pcl::PointCloud<pcl::PointXYZ> test_cloud = createCloud(test_features);

    std::cout << "train cloud: " << train_cloud.points.size() << " points." << std::endl;
    std::cout << "test cloud: " << test_cloud.points.size() << " points." << std::endl;

    pcl::ModelCoefficients coefficients;
    pcl::PointIndices inliers;
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.05);
    seg.setInputCloud(train_cloud.makeShared());
    seg.segment(inliers, coefficients);

    std::cout << "plane model inliers: " << inliers.indices.size () << std::endl;
    double a = coefficients.values[0];
    double b = coefficients.values[1];
    double c = coefficients.values[2];
    double d = coefficients.values[3];

    std::cout << "fitted plane coefficients: " << a << " " << b << " " 
        << c << " " << d << std::endl;

    std::vector<double> distances(test_cloud.points.size());
    for (size_t i = 0; i < test_cloud.points.size(); ++i)
    {
        const pcl::PointXYZ& point = test_cloud.points[i];
        distances[i] = std::abs(a * point.x + b * point.y + c * point.z + d);
    }

    // http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    double m2 = 0; // helper for floating variance computation
    double mean_distance = 0;
    double min_distance = numeric_limits<double>::max();
    double max_distance = 0;
    for (size_t i = 0; i < distances.size(); ++i)
    {
        if (distances[i] > max_distance) max_distance = distances[i];
        if (distances[i] < min_distance) min_distance = distances[i];
        double delta = distances[i] - mean_distance;
        mean_distance += delta / (i + 1);
        m2 += delta * (distances[i] - mean_distance);
    }
    double variance = m2 / (distances.size() - 1);
    double stddev = sqrt(variance);
    std::cout << "distances from fitted plane: mean = " << mean_distance
        << " stddev = " << stddev << " min = " << min_distance 
        << " max = " << max_distance << std::endl;
}


std::vector<TrainTestSet> createSets(
        const std::vector<StereoFeature>& stereo_features, int num_subsets)
{
    std::vector<StereoFeature> sorted_by_size = stereo_features;
    std::sort(sorted_by_size.begin(), sorted_by_size.end(), compareKeyPointSize);

    int subset_size = stereo_features.size() / num_subsets;
    std::vector<std::vector<StereoFeature> > subsets(num_subsets);
    for (size_t i = 0; i < subsets.size(); ++i)
    {
        subsets[i].resize(subset_size);
        std::copy(sorted_by_size.begin() + i * subset_size,
                sorted_by_size.begin() + (i + 1) * subset_size,
                subsets[i].begin());
    }
    std::vector<TrainTestSet> sets(1);
    sets[0].train_features = sorted_by_size;
    sets[0].test_features = sorted_by_size;
    for (size_t i = 0; i < subsets.size(); ++i)
    {
        TrainTestSet equal_set;
        equal_set.train_features = subsets[i];
        equal_set.test_features = subsets[i];
        sets.push_back(equal_set);
    }

    for (size_t i = 0; i < subsets.size(); ++i)
    {
        TrainTestSet big_small_set;
        big_small_set.train_features = sorted_by_size;
        big_small_set.test_features = subsets[i];
        sets.push_back(big_small_set);
    }

    return sets;
}

int main(int argc, char **argv){
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("distance,d", po::value<int>()->required(), "distance to test")
        ("feature_extractor,f", po::value<string>()->default_value("SURF"), "feature extractor")
        ("max_num_key_points,n", po::value<int>()->default_value(10000), "max number of key points in each image")
        ("num_subsets,s", po::value<int>()->default_value(5), "number of subsets to create")
        ;
    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);    
        std::string feature_extractor_name = vm["feature_extractor"].as<std::string>();
        int max_num_key_points = vm["max_num_key_points"].as<int>();
        int distance_to_test = vm["distance"].as<int>();
        int num_subsets = vm["num_subsets"].as<int>();
        
        StereoFeatureExtractor extractor = 
            createStandardExtractor(feature_extractor_name, max_num_key_points);

        cv::Mat left_image, right_image;
        loadImages(distance_to_test, left_image, right_image);

        std::vector<StereoFeature> stereo_features = 
            extractor.extract(left_image, right_image);

        std::cout << "Found " << stereo_features.size() << " stereo features." << std::endl;

        std::vector<TrainTestSet> train_test_sets = 
            createSets(stereo_features, num_subsets);

        std::cout << "running " << train_test_sets.size() << " tests." << std::endl;
        for (size_t i = 0; i < train_test_sets.size(); ++i)
        {
            std::cout << "*** Test " << i << " ***" << std::endl;
            runTest(train_test_sets[i].train_features, 
                    train_test_sets[i].test_features);
            std::cout << std::endl;

        }

    } catch (const po::error& error)
    {
        std::cerr << "Error parsing program options: " << std::endl;
        std::cerr << "  " << error.what() << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    } catch (const std::exception& ex)
    {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return -2;
    } catch (...)
    {
        std::cerr << "Unknown exception caught." << std::endl;
        return -3;
    }
    return 0;
}

