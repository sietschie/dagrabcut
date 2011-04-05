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
    HMM_Component(const HMM_Component& rhs);
    ~HMM_Component();
    HMM_Component& operator=(const HMM_Component& rhs);
};

class HMM : public MM<HMM_Component> {
public:
    void normalize_weights();
    void addModel(const HMM &hmm);
    void addModel(const cv::Mat& gmm, const cv::Mat& compIdxs, const cv::Mat& mask, const cv::Mat& img, int dim = 3);
    void setModel(const cv::Mat& gmm, const cv::Mat& mask, const cv::Mat& img, int dim = 3);
    void setModel(const cv::Mat& gmm);
    void cluster_once();
    HMM() : MM<HMM_Component>() {}
    ~HMM();

    friend void readHMM(const cv::FileNode& fn, HMM& hmm);
    friend cv::FileStorage& operator<<(cv::FileStorage& fs, const HMM& hmm);
};

cv::FileStorage& operator<<(cv::FileStorage& fs, const Gaussian& gauss);
cv::FileStorage& operator<<(cv::FileStorage& fs, const HMM_Component& component);
cv::FileStorage& operator<<(cv::FileStorage& fs, const HMM& hmm);

//cv::FileStorage& operator>>(cv::FileStorage& fs, const Gaussian& gauss);
//cv::FileStorage& operator>>(cv::FileStorage& fs, const HMM_Component& component);
//cv::FileStorage& operator>>(cv::FileNode& fn, HMM& hmm);

void readHMM(const cv::FileNode& fn, HMM& hmm);

#endif //HMM_HPP
