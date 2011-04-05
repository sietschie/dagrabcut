#ifndef MSST_HMM_HPP
#define MSST_HMM_HPP

//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "msst_gmm.hpp"
#include <vector>

//TODO: think about making some stuff private

class MSST_HMM_Component : public MSST_GMM_Component {
public:
    MSST_HMM_Component *left_child, *right_child;
    double div;
    std::vector<std::vector<StructureTensor> > samples;
    std::vector<std::vector<StructureTensor> > get_all_samples();
    MSST_HMM_Component();
//    MSST_HMM_Component(cv::Mat component );
    MSST_HMM_Component(const MSST_HMM_Component& rhs);
    ~MSST_HMM_Component();
    MSST_HMM_Component& operator=(const MSST_HMM_Component& rhs);
};

class MSST_HMM : public MSST_MM<MSST_HMM_Component> {
public:
    void normalize_weights();
    void addModel(const MSST_HMM &hmm);
    void addModel(const cv::Mat& gmm, const cv::Mat& compIdxs, const cv::Mat& mask, const MSStructureTensorImage& img, int dim = 3);
    void setModel(const cv::Mat& gmm, const cv::Mat& mask, const MSStructureTensorImage& img, int dim = 3);
    void cluster_once();
    MSST_HMM() : MSST_MM<MSST_HMM_Component>() {}
    ~MSST_HMM();

    friend void readHMM(const cv::FileNode& fn, MSST_HMM& hmm);
    friend cv::FileStorage& operator<<(cv::FileStorage& fs, const MSST_HMM& hmm);
};

cv::FileStorage& operator<<(cv::FileStorage& fs, const MSST_Gaussian& gauss);
cv::FileStorage& operator<<(cv::FileStorage& fs, const MSST_HMM_Component& component);
cv::FileStorage& operator<<(cv::FileStorage& fs, const MSST_HMM& hmm);

//cv::FileStorage& operator>>(cv::FileStorage& fs, const Gaussian& gauss);
//cv::FileStorage& operator>>(cv::FileStorage& fs, const HMM_Component& component);
//cv::FileStorage& operator>>(cv::FileNode& fn, HMM& hmm);

void readHMM(const cv::FileNode& fn, MSST_HMM& hmm);

#endif //MSST_HMM_HPP
