#include "hmm.hpp"
#include <limits>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace std;

cv::FileStorage& operator<<(cv::FileStorage& fs, const HMM_Component& component)
{
    fs << "{";
    fs << "weight" << component.weight;
    fs << "gauss" << component.gauss;
    fs << "div" << component.div;

    std::vector<cv::Vec3b>::const_iterator itr;
    int size = component.samples.size();
    fs << "nr_samples" << size;
    fs << "samples" << "[";
    for(itr = component.samples.begin(); itr != component.samples.end(); itr++)
    {
        //fs << *itr;
    }
    fs << "]";
    if( component.left_child != NULL )
        fs << "left_child" << *component.left_child;
    if( component.right_child != NULL )
        fs << "right_child" << *component.right_child;
    fs << "}";
    return fs;
}

//cv::FileStorage& operator>>(cv::FileNode& fn, HMM& hmm)
//{
//    int size = (int) (*fn)["nr_components"];
//    int size = (int) fn.container["nr_components"];
//}

void readHMM_Component(const cv::FileNode& fn, HMM_Component& hmmc)
{
    double w = (double)fn["weight"];
    hmmc.weight = w;

    double div = (double)fn["div"];
    hmmc.div = div;

    Gaussian gauss;
    readGaussian(fn["gauss"], gauss);
    hmmc.gauss = gauss;

    if( !fn["left_child"].isNone())
    {
        HMM_Component *hmmc_left = new HMM_Component;
        readHMM_Component(fn["left_child"], *hmmc_left);
        hmmc.left_child = hmmc_left;
    }
    if( !fn["right_child"].isNone())
    {
        HMM_Component *hmmc_right = new HMM_Component;
        readHMM_Component(fn["right_child"], *hmmc_right);
        hmmc.right_child = hmmc_right;
    }

    int size =(int)fn["nr_samples"];
    //std::cout << size << std::endl;

    FileNodeIterator fni = fn["samples"].begin();
    for(int i = 0; i<size; i++)
    {
        Vec3b sample;
        FileNodeIterator sample_fni = (*fni).begin();
        sample[0] = (int) (*sample_fni);
        sample_fni++;
        sample[1] = (int) (*sample_fni);
        sample_fni++;
        sample[2] = (int) (*sample_fni);

        fni++;
        hmmc.samples.push_back(sample);
    }
}

void readHMM(const cv::FileNode& fn, HMM& hmm) {
    int size =(int)fn["nr_components"];
    //std::cout << size << std::endl;

    FileNodeIterator fni = fn["components"].begin();
    for(int i = 0; i<size; i++)
    {
        hmm.setComponentsCount(hmm.getComponentsCount() + 1);
        HMM_Component &hmmc = hmm.components.back();
        readHMM_Component((*fni), hmmc);
        fni++;
    }
}

cv::FileStorage& operator<<(cv::FileStorage& fs, const HMM& hmm)
{
    vector<HMM_Component>::const_iterator itr;

    std::vector<HMM_Component> components;

//    fs << "[";

    fs << "{";// << "CV_PI" << "1+1" << "month" << 12 << "day" << 31 << "year" << 1969 << "g" << "}";
    int size = hmm.components.size();
    fs << "nr_components" << size;
    fs << "components" << "[";
    for(itr = (hmm.components).begin(); itr != hmm.components.end(); itr++)
    {
        fs << *itr;
    }
    //std::cout << "hallo" << endl;
    fs << "]" << "}";
    return fs;
}


void HMM::cluster_once()
{
    vector<HMM_Component>::iterator itr1;
    vector<HMM_Component>::iterator itr2;

    vector<HMM_Component>::iterator min_itr1;
    vector<HMM_Component>::iterator min_itr2;

    double minimum = 1e+100;

    for(itr1 = components.begin(); itr1 != components.end(); itr1++)
        for(itr2 = (itr1 + 1); itr2 != components.end(); itr2++)
        {
            double div = (*itr1).gauss.KLsym((*itr2).gauss);
            if(div < minimum) {
                min_itr1 = itr1;
                min_itr2 = itr2;
                minimum = div;
            }
        }

    HMM_Component hmmc1 = *min_itr1;
    HMM_Component hmmc2 = *min_itr2;

    HMM_Component hmmc_new;
    hmmc_new.left_child = new HMM_Component(hmmc1);
    hmmc_new.right_child = new HMM_Component(hmmc2);

    hmmc_new.weight = hmmc1.weight + hmmc2.weight;
    hmmc_new.div = minimum;


    vector<cv::Vec3b> samples1 = hmmc1.get_all_samples();
    vector<cv::Vec3b> samples2 = hmmc2.get_all_samples();

    samples1.insert(samples1.end(), samples2.begin(), samples2.end());

    hmmc_new.gauss.compute_from_samples(samples1);

    components.erase(min_itr2);
    components.erase(min_itr1);

    components.push_back(hmmc_new);

}

void HMM::add_model(HMM &model)
{
    components.insert(components.end(), model.components.begin(), model.components.end());
}

void HMM::add_model(Mat &model, const Mat &mask, const Mat &img, int dim)
{
    int modelsize = dim /* mean */ + dim * dim /* covariance */ + 1; /* weight */

    components.reserve(model.cols);
    for(int i = 0; i<model.cols; i++)
    {
        HMM_Component hmmc(model.col(i));
        components.push_back(hmmc);
    }

    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == 1 )
            {
                int c = whichComponent( img.at<Vec3b>(p) );
                components[c].samples.push_back(img.at<Vec3b>(p));
            }
        }
    }
}

void HMM::add_model( Mat &model, const Mat &compIdxs, const Mat &mask, const Mat &img, int dim) {
    int modelsize = dim /* mean */ + dim * dim /* covariance */ + 1; /* weight */

    int componentsCount = model.cols / modelsize;

    //std::cout << "componentsCount: " << componentsCount << std::endl;

    double *coefs = model.ptr<double>(0);
    double *mean = coefs + componentsCount;
    double *cov = mean + dim*componentsCount;

    for(int i = 0; i<componentsCount; i++)
    {
        HMM_Component hmmc;
        hmmc.weight = coefs[i];

        double *cpy_mean = mean + (i*dim);
        //std::copy(mean + (i*dim), mean + (i*dim) + dim, cpy_mean);
        hmmc.gauss.mean = Mat(dim, 1, CV_64F);
        for(int j=0; j<dim; j++)
        {
            hmmc.gauss.mean.at<double>(j,0) = cpy_mean[j];
        }
        //hmmc.gauss.mean = Mat(dim, 1, CV_64F, cpy_mean);

        double *cpy_cov = cov + (i*dim*dim);
        //std::copy(cov + (i*dim*dim), cov + (i*dim*dim) + dim*dim, cpy_cov);
        hmmc.gauss.cov = Mat(dim,dim, CV_64F);
        for(int j=0; j<dim; j++)
            for(int k=0; k<dim; k++)
            {
                hmmc.gauss.cov.at<double>(j,k) = cpy_cov[j + k * dim];
            }

        //TODO: muss der speicher wieder freigegeben werden?

        Point p;
        for( p.y = 0; p.y < mask.rows; p.y++ )
        {
            for( p.x = 0; p.x < mask.cols; p.x++ )
            {
                if( mask.at<uchar>(p) == 1 && compIdxs.at<int>(p) == i)
                {
                    hmmc.samples.push_back(img.at<Vec3b>(p));
                }
            }
        }
        components.push_back(hmmc);

    }
}


void HMM::normalize_weights() {
    double sum = 0.0;
    for(int i=0; i<components.size(); i++)
    {
        sum += components[i].weight;
    }

    for(int i=0; i<components.size(); i++)
    {
        components[i].weight /= sum;
    }
}

HMM::~HMM() {
}

vector<cv::Vec3b> HMM_Component::get_all_samples() {
    vector<cv::Vec3b> result(500); //TODO: ist das noetig?

    if( right_child != NULL )
    {
        vector<cv::Vec3b> rsamples = right_child->get_all_samples();
        result.insert(result.end(), rsamples.begin(), rsamples.end());
    }
    if( left_child != NULL )
    {
        vector<cv::Vec3b> lsamples = left_child->get_all_samples();
        result.insert(result.end(), lsamples.begin(), lsamples.end());
    }

    result.insert(result.end(), samples.begin(), samples.end());
    return result;
}

HMM_Component::HMM_Component( Mat Component ) : GMM_Component(Component) {
    left_child = NULL;
    right_child = NULL;
    div = 0;
}

HMM_Component::HMM_Component(const HMM_Component& rhs) : GMM_Component(rhs) {
    left_child = NULL;
    right_child = NULL;
    if(rhs.left_child)
    {
        left_child = new HMM_Component(*rhs.left_child);
    }
    if(rhs.right_child)
    {
        right_child = new HMM_Component(*rhs.right_child);
    }
    samples = rhs.samples;
    div = rhs.div;
}

HMM_Component& HMM_Component::operator=(const HMM_Component& rhs) {
    
    if(this != &rhs)
    {
        HMM_Component* tmp_left = NULL;
        HMM_Component* tmp_right = NULL;
        if(rhs.left_child)
            tmp_left = new HMM_Component(*rhs.left_child);

        if(rhs.right_child)
            tmp_right = new HMM_Component(*rhs.right_child);

        delete left_child;
        delete right_child;

        left_child = tmp_left;
        right_child = tmp_right;

        samples = rhs.samples;
        div = rhs.div;
        gauss = rhs.gauss;
        weight = rhs.weight;
    }
    return *this;
}

HMM_Component::HMM_Component() {
    left_child = NULL;
    right_child = NULL;
    div = 0;
}

HMM_Component::~HMM_Component() {
    delete right_child;
    delete left_child;
}
