#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP

#include <opencv2/core/core.hpp>

class Gaussian {
public:
    cv::Mat mean;
    cv::Mat cov;
    double KLdiv(Gaussian& g2);
    double KLsym(Gaussian& g2);
    Gaussian();
    void compute_from_samples(std::vector<cv::Vec3b> samples);
    ~Gaussian();
};

void readGaussian(const cv::FileNode& fn, Gaussian& gauss);

#endif //GAUSSIAN_HPP


