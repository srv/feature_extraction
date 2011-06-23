#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <cassert>
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

    static const int DEFAULT_MAX_NUM_KEY_POINTS = 200;

    /**
    * Default constructor
    */
    FeatureExtractor() : max_num_key_points_(DEFAULT_MAX_NUM_KEY_POINTS)
    {
    }

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
            cv::Mat& descriptors) = 0;

    /**
    * set the maximum number of key points to extract
    */
    inline void setMaxNumKeyPoints(int num)
    {
        assert(num > 0);
        max_num_key_points_ = num;
    }

    /**
    * \return maximum number of key points
    */
    inline int getMaxNumKeyPoints()
    {
        return max_num_key_points_;
    }

    /// Ptr type
    typedef boost::shared_ptr<FeatureExtractor> Ptr;

  protected:

    int max_num_key_points_;
};

}

#endif


