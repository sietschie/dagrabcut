#ifndef HMM_HPP
#define HMM_HPP

//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

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

class HMM_Component{
public:
    HMM_Component *left_child, *right_child;
    double weight;
    Gaussian gauss;
    std::vector<cv::Vec3b> samples;
    std::vector<cv::Vec3b> get_all_samples();
    HMM_Component();
    ~HMM_Component();
};

class HMM {
public:
    std::vector<HMM_Component*> components;
    //void normalize_weights();
    void add_model(cv::Mat gmm, cv::Mat compIdxs, cv::Mat mask, cv::Mat img, int dim = 3);
    void cluster_once();
    ~HMM();
    double operator()( const cv::Vec3d color ) const;
    double operator()( int ci, const cv::Vec3d color ) const;

};

cv::FileStorage& operator<<(cv::FileStorage& fs, const Gaussian& gauss);
cv::FileStorage& operator<<(cv::FileStorage& fs, const HMM_Component& component);
cv::FileStorage& operator<<(cv::FileStorage& fs, const HMM& hmm);

//cv::FileStorage& operator>>(cv::FileStorage& fs, const Gaussian& gauss);
//cv::FileStorage& operator>>(cv::FileStorage& fs, const HMM_Component& component);
//cv::FileStorage& operator>>(cv::FileNode& fn, HMM& hmm);

void readHMM(const cv::FileNode& fn, HMM& hmm);

#endif //HMM_HPP
