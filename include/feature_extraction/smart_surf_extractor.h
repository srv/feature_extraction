#ifndef SMART_SURF_EXTRACTOR_H
#define SMART_SURF_EXTRACTOR_H

#include <opencv2/features2d/features2d.hpp>
#include "feature_extractor.h"

#include "feature_extraction/smart_surf.h"

namespace feature_extraction
{

/**
* \class SmartSurfExtractor
* \brief Extractor that uses our own implementation of SURF
*/
class SmartSurfExtractor : public FeatureExtractor
{

  public:
    /**
    * Constructs a surf extractor with default parameters.
    */
    SmartSurfExtractor();

    // see base class for documentation
    void extract(const cv::Mat& image, std::vector<KeyPoint>& key_points,
            cv::Mat& descriptors, const cv::Rect& roi);

  private:

    visual_odometry::SmartSurf surf_;
};

}

#endif

