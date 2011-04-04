#ifndef STRUCTURETENSOR_HPP
#define STRUCTURETENSOR_HPP

#include <opencv2/core/core.hpp>
#include <vector>

class StructureTensor{
public:
    StructureTensor();
    StructureTensor(const cv::Vec3d& t);
    StructureTensor(const cv::Mat& m);
    StructureTensor(double p00, double p11, double p01_10);
    cv::Mat getMatrix() const;
private:
    cv::Mat st;
};

// Multiscale
class MSStructureTensorImage {
public:
    MSStructureTensorImage(const cv::Mat& image);
    MSStructureTensorImage() {}
    std::vector<StructureTensor> getTensor(int x, int y) const;
    std::vector<std::vector<StructureTensor> > getAllTensors() const;
    int cols, rows;
protected:
    std::vector<std::vector<StructureTensor> > tensors;
};

class StructureTensorImage {
public:
/**
 * Constructor that reads an image and creates the structure tensor for the image
 *
 * @param image the input image
 */
    StructureTensorImage(const cv::Mat& image, double sigma = 8.0);

/**
 * Get structure tensor from pixel position x, y
 *
 * @param x x coordinate to get the tensor from
 * @param y y coordinate to get the tensor from
 */
    StructureTensor getTensor(int x, int y) const;
    std::vector<StructureTensor> getAllTensors() const;
    cv::Mat getImage();
    int cols, rows;

private:
    std::vector<StructureTensor> tensors;
    cv::Mat blurredstmat;
};

double MSST_kmeans(const std::vector<std::vector<StructureTensor> > &tensors, int K, cv::TermCriteria criteria, int attempts, cv::Mat& best_labels, 
std::vector<std::vector<StructureTensor> >& best_centers);

double kmeans(const std::vector<StructureTensor> &tensors, int K, cv::TermCriteria criteria, int attempts, cv::Mat& best_labels, std::vector<StructureTensor>& best_centers);


/**
 * Computes the symmetric KL distance between two structure tensors 
 *
 * @param l first structure tensor
 * @param r second structure tensor
 */
double distance2(const StructureTensor& l, const StructureTensor& r);
double MS_distance2(const std::vector<StructureTensor>& l, const std::vector<StructureTensor>& r);

/**
 * Computes the mean structure tensor given a list of structure tensors 
 *
 * @param list vector of structure tensors
 */
StructureTensor compute_mean(const std::vector<StructureTensor>& list);
std::vector<StructureTensor> MS_compute_mean(const std::vector<std::vector<StructureTensor> >& list);

/**
 * Computes the variance given a list of structure tensors 
 *
 * @param list vector of structure tensors
 */
double compute_variance(const std::vector<StructureTensor>& list);
double MS_compute_variance(const std::vector<std::vector<StructureTensor> >& list);

#endif //STRUCTURETENSOR_HPP
