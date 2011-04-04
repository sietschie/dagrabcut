#ifndef ST_GAUSSIAN_HPP
#define ST_GAUSSIAN_HPP

#include <opencv2/core/core.hpp>
#include "structuretensor.hpp"

class ST_Gaussian {
public:
    cv::Mat mean;
    cv::Mat cov;
    double KLdiv(const ST_Gaussian& g2);
    double KLsym(const ST_Gaussian& g2);
    ST_Gaussian();
    ST_Gaussian(const ST_Gaussian& rhs);
    void compute_from_samples(std::vector<StructureTensor> samples);
    ~ST_Gaussian();
    ST_Gaussian& operator=(const ST_Gaussian& rhs);
};

void readGaussian(const cv::FileNode& fn, ST_Gaussian& gauss);

class MSST_Gaussian {
public:
    std::vector<StructureTensor> mean;
    double cov;
    double KLdiv(const MSST_Gaussian& g2);
    double KLsym(const MSST_Gaussian& g2);
    void compute_from_samples(std::vector<std::vector<StructureTensor> > samples);
};


#endif //ST_GAUSSIAN_HPP


