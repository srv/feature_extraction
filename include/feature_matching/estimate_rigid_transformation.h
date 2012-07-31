namespace feature_matching
{

/**
 * Estimates a rigid transformation between two matching 3D point sets.
 */
bool estimateRigidTransformation(
    const std::vector<cv::Point3d> source_points, 
    const std::vector<cv::Point3d> target_points,
    cv::Mat& transform, std::vector<unsigned char>& inliers,
    double inlier_threshold, double min_sample_distance, int max_iterations = 1000);

}
