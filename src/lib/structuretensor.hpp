#ifndef STRUCTURETENSOR_HPP
#define STRUCTURETENSOR_HPP

#include <opencv2/core/core.hpp>
#include <vector>

class StructureTensorImage {
/**
 * Constructor that reads an image and creates the structure tensor for the image
 *
 * @param image the input image
 */
    StructureTensorImage(const cv::Mat& image);

/**
 * Computes the symmetric KL distance between two structure tensors 
 *
 * @param l first structure tensor
 * @param r second structure tensor
 */
    double distance(const cv::Vec3d& l, const cv::Vec3d& r);

/**
 * Computes the mean structure tensor given a list of structure tensors 
 *
 * @param list vector of structure tensors
 */
    cv::Vec3d mean(const std::vector<cv::Vec3d>& list);

/**
 * Computes the variance given a list of structure tensors 
 *
 * @param list vector of structure tensors
 */
    cv::Vec3d variance(const std::vector<cv::Vec3d>& list);
};

#endif //STRUCTURETENSOR_HPP
