#include "stereo_feature.h"

std::ostream& operator<<(std::ostream& ostream,
        const stereo_feature_extraction::StereoFeature& feature)
{
    ostream << "wp: " << feature.world_point
            << " kp: " << feature.key_point.pt
            << " color: [" << (int)feature.color[0] << "," 
                           << (int)feature.color[1] << "," 
                           << (int)feature.color[2] << "]"
            << " desc: " << feature.descriptor.at<float>(0, 0);
    return ostream;
}


