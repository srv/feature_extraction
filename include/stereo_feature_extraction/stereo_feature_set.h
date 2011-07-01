#ifndef STEREO_FEATURE_SET_H
#define STEREO_FEATURE_SET_H

#include "stereo_feature.h"

namespace stereo_feature_extraction
{

/**
* A set of stereo features with convenience methods
*/
class StereoFeatureSet
{
    public:
        /**
        * Saves the points as point cloud data file (see
        * http://pointclouds.org/documentation/tutorials/pcd_file_format.php
        * for a description of the format.
        */
        void savePointCloud(const std::string& file_name);

        /**
        * Saves the feature descriptors as a point cloud data file
        */
        void saveFeatureCloud(const std::string& file_name);

        /**
        * Holds the features
        */
        std::vector<StereoFeature> stereo_features;
};

}

#endif
