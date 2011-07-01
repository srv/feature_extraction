#include <fstream>

#include "stereo_feature_set.h"


std::ostream& operator<<(std::ostream& ostream,
        const stereo_feature_extraction::StereoFeatureSet& feature_set)
{
    ostream << "Stereo feature set with " << feature_set.stereo_features.size()
        << " features: " << std::endl;
    for (size_t i = 0; i < feature_set.stereo_features.size(); ++i)
    {
        ostream << feature_set.stereo_features[i] << std::endl;
    }
    return ostream;
}

namespace stereo_feature_extraction
{

float packRgb_(const cv::Vec3b& color)
{
    int32_t rgb = (color[2] << 16) | (color[1] << 8) | color[0];
    return rgb;
}

bool StereoFeatureSet::savePointCloud(const std::string& file_name)
{
    std::ofstream out(file_name.c_str());
    if (!out.is_open())
    {
        std::cerr << "Error: StereoFeatureSet::savePointCloud(): "
            " Cannot open file '" << file_name << "' for writing!" << std::endl;
        return false;
    }
    out << "# Stereo Feature 3D points" << std::endl;
    out << "FIELDS x y z rgb" << std::endl;
    out << "SIZE 4 4 4 4" << std::endl;
    out << "TYPE F F F F" << std::endl;
    out << "COUNT 1 1 1 1" << std::endl;
    out << "WIDTH " << stereo_features.size() << std::endl;
    out << "HEIGHT 1" << std::endl;
    out << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
    out << "POINTS " << stereo_features.size() << std::endl;
    out << "DATA ascii" << std::endl;
    for (size_t i = 0; i < stereo_features.size(); ++i)
    {
        out << stereo_features[i].world_point.x << " ";
        out << stereo_features[i].world_point.y << " ";
        out << stereo_features[i].world_point.z << " ";
        out << packRgb_(stereo_features[i].color) << std::endl;
    }
    out.close();
    return true;
}

bool StereoFeatureSet::saveFeatureCloud(const std::string& file_name)
{
    std::ofstream out(file_name.c_str());
    if (!out.is_open())
    {
        std::cerr << "Error: StereoFeatureSet::saveFeatureCloud(): "
            " Cannot open file '" << file_name << "' for writing!" << std::endl;
        return false;
    }
    
    int descriptor_size; 
    if (stereo_features.size() == 0)
    {
        descriptor_size = 0;
    }
    else
    {
        descriptor_size = stereo_features[0].descriptor.cols;
    }
    out << "# Stereo Feature Descriptors" << std::endl;
    out << "FIELDS data" << std::endl;
    out << "SIZE 4" << std::endl;
    out << "TYPE F" << std::endl;
    out << "COUNT " << descriptor_size << std::endl;
    out << "WIDTH " << stereo_features.size() << std::endl;
    out << "HEIGHT 1" << std::endl;
    out << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
    out << "POINTS " << stereo_features.size() << std::endl;
    out << "DATA ascii" << std::endl;
    for (size_t i = 0; i < stereo_features.size(); ++i)
    {
        cv::Mat& descriptor = stereo_features[i].descriptor;
        if (descriptor.rows == 1)
        {
            for (int c = 0; c < descriptor.cols; ++c)
            {
                out << descriptor.at<float>(0, c) << " ";
            }
        }
        out << std::endl;
    }
    out.close();
    return true;
}

}

