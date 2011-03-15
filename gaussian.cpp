#include "gaussian.hpp"
#include <iostream>

using namespace cv;
using namespace std;

cv::FileStorage& operator<<(cv::FileStorage& fs, const Gaussian& gauss)
{
    fs << "{";
    fs << "mean" << gauss.mean;
    fs << "cov" << gauss.cov;
    fs << "}";
    return fs;
}

void readGaussian(const cv::FileNode& fn, Gaussian& gauss)
{
    fn["mean"] >> gauss.mean;
    fn["cov"] >> gauss.cov;
}

double Gaussian::KLdiv(Gaussian &g2) {
    //  1/2 * (trace(cov2_inv cov1) + (mean2 - mean1)^T cov2_inv (mean2 - mean1) - log_e (det_cov1 / det_cov2) - dim
    // source: wikipedia, jMEF

    Mat mean_diff = g2.mean - mean;
    Mat cov2_inv = g2.cov.inv();
    double det_cov1 = determinant(cov);
    double det_cov2 = determinant(g2.cov);
    double log_e_det_cov1_det_cov2 = log(det_cov1 / det_cov2);
    double trace_cov2_inv_cov1 = (trace( cov2_inv * cov ))[0];

    Mat mean_diff_transposed;
    transpose(mean_diff, mean_diff_transposed);

    double tmp = Mat(mean_diff_transposed * cov2_inv * mean_diff).at<double>(0,0);

    double kldiv = ( trace_cov2_inv_cov1 + tmp - log_e_det_cov1_det_cov2 - cov.cols ) / 2.0;
    return kldiv;
}

double Gaussian::KLsym(Gaussian &g2) {
    // (mean2 - mean1)^T (cov1_inv + cov2_inv) (mean2 - mean1) + trace( cov1_inv cov2 + cov1 cov2_inv ) - 2dim
    // from http://www.sciweavers.org/files/docs/2358/icassp_cvd_poster_pdf_4a383d1fb0.pdf

    /*    std::cout << "entered KLsym" << std::endl;
        std::cout << mean << std::endl;
        std::cout << g2.mean << std::endl;
        std::cout << cov << std::endl;
        std::cout << g2.cov << std::endl;
    */
    Mat mean_diff = g2.mean - mean;
    Mat mean_diff_transposed;
    transpose(mean_diff, mean_diff_transposed);

    Mat cov1_inv = cov.inv();
    Mat cov2_inv = g2.cov.inv();

    double summand1 = Mat(mean_diff_transposed * (cov1_inv + cov2_inv) * mean_diff).at<double>(0,0);
    double summand2 = trace( cov1_inv * g2.cov + cov * cov2_inv )[0];

//    std::cout << "mean_diff: " << mean_diff << "  cov1_inv: " << cov1_inv << "  cov2_inv: " << cov2_inv << std::endl;
//    std::cout << "summand1: " << summand1 << "  summand2: " << summand2 << "  sum: " << summand1 + summand2 << std::endl;

    double klsym = summand1 + summand2 - 2 * cov.cols;

    return klsym / 2.0;
}

Gaussian::Gaussian() {}

void Gaussian::compute_from_samples(vector<cv::Vec3b> samples) {
    mean = Mat::zeros(3, 1, CV_64F);
    cov = Mat::zeros(3, 3, CV_64F);

    vector<cv::Vec3b>::iterator itr;

    for(itr=samples.begin(); itr!=samples.end(); itr++)
    {
        mean.at<double>(0,0) += (*itr)[0];
        mean.at<double>(1,0) += (*itr)[1];
        mean.at<double>(2,0) += (*itr)[2];

        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                cov.at<double>(i,j) += (*itr)[i] * (*itr)[j];
    }

    mean = mean / samples.size();
    cov = cov / samples.size();

    //std::cout << samples.size() << std::endl;
    //std::cout << cov << std::endl;

    if(samples.size() == 0) //TODO: wie richtig behandeln?
    {
        for(int i=0; i<3; i++)
        {
            mean.at<double>(i,0) = 0.0;
            for(int j=0; j<3; j++)
                cov.at<double>(i,j) = 0.0;
        }

    }

    if( determinant(cov) < std::numeric_limits<double>::epsilon() )
    {
        // Adds the white noise to avoid singular covariance matrix.
        double variance = 0.01;
        cov.at<double>(0,0) += variance;
        cov.at<double>(1,1)  += variance;
        cov.at<double>(2,2)  += variance;
    }
}

Gaussian::~Gaussian() {
//    delete mean.ptr<double>(0);
//    delete cov.ptr<double>(0);
}


