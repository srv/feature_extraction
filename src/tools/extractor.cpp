#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/stereo_camera_model.h>
#include <feature_extraction/feature_extractor_factory.h>

#include "stereo_feature_extraction/stereo_feature_extractor.h"
#include "stereo_feature_extraction/drawing.h"


using namespace stereo_feature_extraction;
using namespace feature_extraction;
namespace po = boost::program_options;

int main(int argc, char** argv)
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("ileft,L", po::value<std::string>()->required(), "left image file")
        ("iright,R", po::value<std::string>()->required(), "right image file")
        ("cleft,J", po::value<std::string>()->required(), 
                "calibration file of left camera")
        ("cright,K", po::value<std::string>()->required(), 
                "calibration file of right camera")
        ("max_y_diff,Y", po::value<double>()->default_value(2.0), "maximum y difference for matching keypoints")
        ("max_angle_diff,A", po::value<double>()->default_value(4.0), "maximum angle difference for matching keypoints")
        ("max_size_diff,S", po::value<int>()->default_value(5), "maximum size difference for matching keypoints")
        ("feature_extractor,E", po::value<std::string>()->default_value("SURF"), "feature extractor")
        ("min_depth,N", po::value<double>()->default_value(0.0), "minimum depth")
        ("max_depth,F", po::value<double>()->default_value(100.0), "maximum depth")
        ("max_num_key_points,M", po::value<int>()->default_value(5000), 
                "maximum number of key points to extract")
        ("cloud_file,C", po::value<std::string>(), "file name for output point cloud")
        ("descriptor_file,D", po::value<std::string>(), "file name for output descriptors")
        ("verbose,V", "vebose output")
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
        return -1;
    }

    // extract params
    std::string left_image_file = vm["ileft"].as<std::string>();
    std::string right_image_file = vm["iright"].as<std::string>();
    std::string left_calibration_file = vm["cleft"].as<std::string>();
    std::string right_calibration_file = vm["cright"].as<std::string>();
    double max_y_diff = vm["max_y_diff"].as<double>();
    double max_angle_diff = vm["max_angle_diff"].as<double>();
    int max_size_diff = vm["max_size_diff"].as<int>();
    std::string feature_extractor_name = vm["feature_extractor"].as<std::string>();
    double min_depth = vm["min_depth"].as<double>();
    double max_depth = vm["max_depth"].as<double>();
    int max_num_key_points = vm["max_num_key_points"].as<int>();

    bool verbose = vm.count("verbose") > 0;

    // create instances
    FeatureExtractor::Ptr feature_extractor = 
        FeatureExtractorFactory::create(feature_extractor_name);
    if (feature_extractor.get() == 0)
    {
        std::cerr << "Cannot create feature extractor with name '" 
            << feature_extractor_name << "'" << std::endl;
        return -2;
    }

    StereoCameraModel::Ptr stereo_camera_model = 
        StereoCameraModel::Ptr(new StereoCameraModel());
    stereo_camera_model->fromCalibrationFiles(left_calibration_file, right_calibration_file);

        StereoFeatureExtractor extractor;

    // set variables
    feature_extractor->setMaxNumKeyPoints(max_num_key_points);
    extractor.setFeatureExtractor(feature_extractor);
    extractor.setCameraModel(stereo_camera_model);
    extractor.setMaxYDiff(max_y_diff);
    extractor.setMaxAngleDiff(max_angle_diff);
    extractor.setMaxSizeDiff(max_size_diff);
    extractor.setMinDepth(min_depth);
    extractor.setMaxDepth(max_depth);

    if (verbose) std::cout << "Running extractor..." << std::flush;
    cv::Mat image_left = cv::imread(left_image_file);
    cv::Mat image_right = cv::imread(right_image_file);

    StereoFeatureSet stereo_feature_set = 
        extractor.extract(image_left, image_right);
    if (verbose) std::cout << "found " << stereo_feature_set.stereo_features.size() 
        << " stereo features." << std::endl;

    if (vm.count("display"))
    {
        cv::Mat result_image;
        drawStereoFeatures(result_image, image_left, image_right, 
                stereo_feature_set.stereo_features);

        cv::namedWindow("matchings", CV_WINDOW_NORMAL);
        cv::imshow("matchings", result_image);
        cvWaitKey();
    }

    if (vm.count("cloud_file"))
    {
        std::string file_name = vm["cloud_file"].as<std::string>();
        if (stereo_feature_set.savePointCloud(file_name))
        {
            if (verbose) std::cout << "Written point cloud to file '" << file_name << "'." 
                << std::endl;
        }
        else
        {
            std::cerr << "Could not write points." << std::endl;
        }
    }

    if (vm.count("descriptor_file"))
    {
        std::string file_name = vm["descriptor_file"].as<std::string>();
        if (stereo_feature_set.saveFeatureCloud(file_name))
        {
            if (verbose) std::cout << "Written descriptors to file '" << file_name << "'." 
                 << std::endl;
        }
        else
        {
            std::cerr << "Could not save features." << std::endl;
        }
    }    return 0; 
}

