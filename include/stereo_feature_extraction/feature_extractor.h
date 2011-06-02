#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <boost/shared_ptr.hpp>

#include <opencv2/features2d/features2d.hpp>

#include "key_point.h"

namespace stereo_feature_extraction
{

/**
* \class FeatureExtractor
* \brief Common interface for feature extractors
*/
class FeatureExtractor
{

  public:
    /**
    * Virtual destructor
    */
    virtual ~FeatureExtractor() {};

    /**
    * Extraction interface.
    * \param image the input image.
    * \param mask mask out keypoints where mask.at<uchar>(i, j) == 0
    * \param key_points output vector for computed key points
    * \param descriptors output matrix for descriptors 
    */
    virtual void extract(const cv::Mat& image, 
            const cv::Mat& mask,
            std::vector<KeyPoint>& key_points,
            cv::Mat& descriptors) const = 0;

    /// Ptr type
    typedef boost::shared_ptr<FeatureExtractor> Ptr;
};

}

#endif


