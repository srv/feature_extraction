#ifndef SURF_EXTRACTOR_H
#define SURF_EXTRACTOR_H

#include <opencv2/features2d/features2d.hpp>
#include "feature_extractor.h"

#include "Surf.h"

namespace stereo_feature_extraction
{

/**
* \class SurfExtractor
* \brief Extractor that uses our own implementation of SURF
*/
class SurfExtractor : public FeatureExtractor
{

  public:
    /**
    * Constructs a surf extractor with default parameters.
    */
    SurfExtractor();

    // see base class for documentation
    void extract(const cv::Mat& image, const cv::Mat& mask,
            std::vector<KeyPoint>&, cv::Mat& descriptors);

  private:

    odometry::Surf surf_;
};

}

#endif

