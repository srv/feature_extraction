#ifndef CV_SURF_EXTRACTOR_H
#define CV_SURF_EXTRACTOR_H

#include <opencv2/features2d/features2d.hpp>
#include "feature_extractor.h"

namespace stereo_feature_extraction
{

/**
* \class CvSurfExtractor
* \brief Extractor that uses openCV's implementation of SURF
*/
class CvSurfExtractor : public FeatureExtractor
{

  public:
    /**
    * Constructs a surf extractor with default parameters.
    */
    CvSurfExtractor();

    // see base class for documentation
    void extract(const cv::Mat& image, const cv::Mat& mask,
            std::vector<KeyPoint>&, cv::Mat& descriptors) const;

  private:

    // holds opencv's surf method
    cv::SURF surf_;
};

}

#endif

