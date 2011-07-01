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
        bool savePointCloud(const std::string& file_name);

        /**
        * Saves the feature descriptors as a point cloud data file
        */
        bool saveFeatureCloud(const std::string& file_name);

        /**
        * Holds the features
        */
        std::vector<StereoFeature> stereo_features;
};

}

std::ostream& operator<<(std::ostream& ostream,
        const stereo_feature_extraction::StereoFeatureSet& feature_set);


#endif
