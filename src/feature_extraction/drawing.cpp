#include <opencv2/core/core.hpp>

#include "feature_extraction/drawing.h"
#include "feature_extraction/key_point.h"

namespace feature_extraction
{

    void drawKeyPoint(cv::Mat& image, const KeyPoint& key_point)
    {
        cv::Point center(cvRound(key_point.pt.x), 
                        cvRound(key_point.pt.y));
        int radius = cvRound(key_point.size) / 2;
        cv::circle(image, center, radius, cv::Scalar(0, 255, 0), 2);
    }

    void drawKeyPoints(cv::Mat& image, const std::vector<KeyPoint>& key_points)
    {
        for (size_t i = 0; i < key_points.size(); ++i)
        {
            drawKeyPoint(image, key_points[i]);
        }
    }
}
