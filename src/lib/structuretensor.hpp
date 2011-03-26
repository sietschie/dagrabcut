#ifndef STRUCTURETENSOR_HPP
#define STRUCTURETENSOR_HPP

#include <opencv2/core/core.hpp>
#include <vector>

class StructureTensor{
public:
    StructureTensor(const cv::Vec3d& t);
    cv::Mat getMatrix();
private:
    cv::Mat st;
};

class StructureTensorImage {
public:
/**
 * Constructor that reads an image and creates the structure tensor for the image
 *
 * @param image the input image
 */
    StructureTensorImage(const cv::Mat& image, double sigma = 1.0);

/**
 * Get structure tensor from pixel position x, y
 *
 * @param x x coordinate to get the tensor from
 * @param y y coordinate to get the tensor from
 */
    StructureTensor getTensor(int x, int y);
    cv::Mat getImage();

private:
    std::vector<StructureTensor> tensors;
    cv::Mat blurredstmat;
};


/**
 * Computes the symmetric KL distance between two structure tensors 
 *
 * @param l first structure tensor
 * @param r second structure tensor
 */
double distance(StructureTensor& l, StructureTensor& r);

/**
 * Computes the mean structure tensor given a list of structure tensors 
 *
 * @param list vector of structure tensors
 */
StructureTensor mean(std::vector<StructureTensor>& list);

/**
 * Computes the variance given a list of structure tensors 
 *
 * @param list vector of structure tensors
 */
StructureTensor variance(const std::vector<StructureTensor>& list);

#endif //STRUCTURETENSOR_HPP
