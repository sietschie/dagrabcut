#ifndef _SHARED_H_
#define _SHARED_H_

#include <string>
#include <opencv2/core/core.hpp>
#include "structuretensor.hpp"

void readImageAndMask(std::string filename, cv::Mat& image, cv::Mat& mask);

double compute_variance(std::vector<std::string> input_images, cv::Mat mean_bgdModel, cv::Mat mean_fgdModel, int nr_gaussians, int class_number,
        double &var_bgd_kl_sym, double &var_bgd_kl_mr, double &var_bgd_kl_rm, double &var_fgd_kl_sym, double &var_fgd_kl_mr, double &var_fgd_kl_rm );

void learnGMMfromSamples(std::vector<cv::Vec3f> samples, cv::Mat& model, int nr_gaussians = 5);

double compute_variance_from_vector(std::vector<double> diffs);

double compute_probability(double dist, double variance);


double MSST_compute_variance(std::vector<std::string> input_images, cv::Mat mean_bgdModel, cv::Mat mean_fgdModel, int nr_gaussians, int class_number,
        double &var_bgd_kl_sym, double &var_bgd_kl_mr, double &var_bgd_kl_rm, double &var_fgd_kl_sym, double &var_fgd_kl_mr, double &var_fgd_kl_rm );

void MSST_learnGMMfromSamples(const std::vector<std::vector<StructureTensor> > &samples, cv::Mat& model, int nr_gaussians = 5);


#endif /* _SHARED_H_ */

