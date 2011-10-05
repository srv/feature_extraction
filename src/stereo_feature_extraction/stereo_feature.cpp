#include "stereo_feature_extraction/stereo_feature.h"

std::ostream& operator<<(std::ostream& ostream,
        const stereo_feature_extraction::StereoFeature& feature)
{
    ostream << "wp: " << feature.world_point
            << " kp left: " << feature.key_point_left.pt
            << " kp right: " << feature.key_point_right.pt
            << " color: [" << (int)feature.color[0] << "," 
                           << (int)feature.color[1] << "," 
                           << (int)feature.color[2] << "]"
            << " desc: [";
    if (feature.descriptor.data != NULL)
    {
        // print out some descriptor data
        for (int i = 0; i < feature.descriptor.cols && i < 4; ++i)
        {
            ostream << feature.descriptor.at<float>(0, i);
        }
        ostream << "...";
    }
    else
    {
        ostream << "<empty>";
    }
    ostream << "]";
    return ostream;
}


