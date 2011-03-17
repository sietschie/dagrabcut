#ifndef HMM_HPP
#define HMM_HPP

//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "gmm.hpp"
#include <vector>

//TODO: think about making some stuff private

class HMM_Component : public GMM_Component {
public:
    HMM_Component *left_child, *right_child;
    double div;
    std::vector<cv::Vec3b> samples;
    std::vector<cv::Vec3b> get_all_samples();
    HMM_Component(cv::Mat component );
    HMM_Component();
    ~HMM_Component();
};

class HMM {
public:
    std::vector<HMM_Component*> components;
    void normalize_weights();
    void add_model(HMM &hmm);
    void add_model(cv::Mat& gmm, const cv::Mat& compIdxs, const cv::Mat& mask, const cv::Mat& img, int dim = 3);
    void add_model(cv::Mat& gmm, const cv::Mat& mask, const cv::Mat& img, int dim = 3);
    cv::Mat get_model();
    void cluster_once();
    ~HMM();
    double operator()( const cv::Vec3d color ) const;
    double operator()( int ci, const cv::Vec3d color ) const;
    int whichComponent( const cv::Vec3d color ) const;

    double KLdiv(const HMM& rhs);
    double KLsym(HMM& rhs);
    void free_components();

};

cv::FileStorage& operator<<(cv::FileStorage& fs, const Gaussian& gauss);
cv::FileStorage& operator<<(cv::FileStorage& fs, const HMM_Component& component);
cv::FileStorage& operator<<(cv::FileStorage& fs, const HMM& hmm);

//cv::FileStorage& operator>>(cv::FileStorage& fs, const Gaussian& gauss);
//cv::FileStorage& operator>>(cv::FileStorage& fs, const HMM_Component& component);
//cv::FileStorage& operator>>(cv::FileNode& fn, HMM& hmm);

void readHMM(const cv::FileNode& fn, HMM& hmm);

#endif //HMM_HPP