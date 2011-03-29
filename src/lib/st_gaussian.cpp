#include "st_gaussian.hpp"
#include <iostream>

using namespace cv;
using namespace std;

ST_Gaussian& ST_Gaussian::operator=(const ST_Gaussian& rhs)
{
    if(this != &rhs)
    {
        cov = rhs.cov.clone();
        mean = rhs.mean.clone();
    }
    return *this;
}

ST_Gaussian::ST_Gaussian(const ST_Gaussian& rhs)
{
    cov = rhs.cov.clone();
    mean = rhs.mean.clone();
}


cv::FileStorage& operator<<(cv::FileStorage& fs, const ST_Gaussian& gauss)
{
    fs << "{";
    fs << "mean" << gauss.mean;
    fs << "cov" << gauss.cov;
    fs << "}";
    return fs;
}

void readGaussian(const cv::FileNode& fn, ST_Gaussian& gauss)
{
    fn["mean"] >> gauss.mean;
    fn["cov"] >> gauss.cov;
}

double ST_Gaussian::KLdiv(const ST_Gaussian &g2) {
    //  1/2 * (trace(cov2_inv cov1) + (mean2 - mean1)^T cov2_inv (mean2 - mean1) - log_e (det_cov1 / det_cov2) - dim
    // source: wikipedia, jMEF

/*    Mat mean_diff = g2.mean - mean;
    Mat cov2_inv = g2.cov.inv();
    double det_cov1 = determinant(cov);
    double det_cov2 = determinant(g2.cov);
    double log_e_det_cov1_det_cov2 = log(det_cov1 / det_cov2);
    double trace_cov2_inv_cov1 = (trace( cov2_inv * cov ))[0];

    Mat mean_diff_transposed;
    transpose(mean_diff, mean_diff_transposed);

    double tmp = Mat(mean_diff_transposed * cov2_inv * mean_diff).at<double>(0,0);

    double kldiv = ( trace_cov2_inv_cov1 + tmp - log_e_det_cov1_det_cov2 - cov.cols ) / 2.0;
    return kldiv;*/
    
    assert(1);
    return 0.0;
}

double ST_Gaussian::KLsym(const ST_Gaussian &g2) {
    // (mean2 - mean1)^T (cov1_inv + cov2_inv) (mean2 - mean1) + trace( cov1_inv cov2 + cov1 cov2_inv ) - 2dim
    // from http://www.sciweavers.org/files/docs/2358/icassp_cvd_poster_pdf_4a383d1fb0.pdf

    /*    std::cout << "entered KLsym" << std::endl;
        std::cout << mean << std::endl;
        std::cout << g2.mean << std::endl;
        std::cout << cov << std::endl;
        std::cout << g2.cov << std::endl;
    */
/*    Mat mean_diff = g2.mean - mean;
    Mat mean_diff_transposed;
    transpose(mean_diff, mean_diff_transposed);

    Mat cov1_inv = cov.inv();
    Mat cov2_inv = g2.cov.inv();

    double summand1 = Mat(mean_diff_transposed * (cov1_inv + cov2_inv) * mean_diff).at<double>(0,0);
    double summand2 = trace( cov1_inv * g2.cov + cov * cov2_inv )[0];

//    std::cout << "mean_diff: " << mean_diff << "  cov1_inv: " << cov1_inv << "  cov2_inv: " << cov2_inv << std::endl;
//    std::cout << "summand1: " << summand1 << "  summand2: " << summand2 << "  sum: " << summand1 + summand2 << std::endl;

    double klsym = summand1 + summand2 - 2 * cov.cols;

    return klsym / 2.0;*/

    assert(1);
    return 0.0;
}

ST_Gaussian::ST_Gaussian() {}

void ST_Gaussian::compute_from_samples(vector<StructureTensor> samples) {
    std::cout << "samples.size() = " << samples.size() << std::endl;
    mean = Mat::zeros(2, 2, CV_64F);
    cov = Mat::zeros(1, 1, CV_64F);

    mean = compute_mean(samples).getMatrix();
    cov.at<double>(0,0) = compute_variance(samples);

    assert(cov.at<double>(0,0) != 0.0);

/*    if( determinant(cov) < std::numeric_limits<double>::epsilon() )
    {
        // Adds the white noise to avoid singular covariance matrix.
        double variance = 0.01;
        cov.at<double>(0,0) += variance;
        cov.at<double>(1,1)  += variance;
        cov.at<double>(2,2)  += variance;
    }*/
}

ST_Gaussian::~ST_Gaussian() {
//    delete mean.ptr<double>(0);
//    delete cov.ptr<double>(0);
}


