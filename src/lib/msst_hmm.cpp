#include "msst_hmm.hpp"
#include <limits>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace std;

cv::FileStorage& operator<<(cv::FileStorage& fs, const MSST_HMM_Component& component)
{
    fs << "{";
    fs << "weight" << component.weight;
    fs << "gauss" << component.gauss;
    fs << "div" << component.div;

//    std::vector<cv::Vec3b>::const_iterator itr;
    int size = component.samples.size();
    fs << "nr_samples" << size;
/*    fs << "samples" << "[";
    for(itr = component.samples.begin(); itr != component.samples.end(); itr++)
    {
        //fs << *itr;
    }
    fs << "]";*/
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

void readHMM_Component(const cv::FileNode& fn, MSST_HMM_Component& hmmc)
{
    double w = (double)fn["weight"];
    hmmc.weight = w;

    double div = (double)fn["div"];
    hmmc.div = div;

    MSST_Gaussian gauss;
    readGaussian(fn["gauss"], gauss);
    hmmc.gauss = gauss;

    if( !fn["left_child"].isNone())
    {
        MSST_HMM_Component *hmmc_left = new MSST_HMM_Component;
        readHMM_Component(fn["left_child"], *hmmc_left);
        hmmc.left_child = hmmc_left;
    }
    if( !fn["right_child"].isNone())
    {
        MSST_HMM_Component *hmmc_right = new MSST_HMM_Component;
        readHMM_Component(fn["right_child"], *hmmc_right);
        hmmc.right_child = hmmc_right;
    }

    int size =(int)fn["nr_samples"];
    //std::cout << size << std::endl;

/*    FileNodeIterator fni = fn["samples"].begin();
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
    }*/
}

void readHMM(const cv::FileNode& fn, MSST_HMM& hmm) {
    int size =(int)fn["nr_components"];
    //std::cout << size << std::endl;

    FileNodeIterator fni = fn["components"].begin();
    for(int i = 0; i<size; i++)
    {
        hmm.setComponentsCount(hmm.getComponentsCount() + 1);
        MSST_HMM_Component &hmmc = hmm.components.back();
        readHMM_Component((*fni), hmmc);
        fni++;
    }
}

cv::FileStorage& operator<<(cv::FileStorage& fs, const MSST_HMM& hmm)
{
    vector<MSST_HMM_Component>::const_iterator itr;

    std::vector<MSST_HMM_Component> components;

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


void MSST_HMM::cluster_once()
{
    vector<MSST_HMM_Component>::iterator itr1;
    vector<MSST_HMM_Component>::iterator itr2;

    vector<MSST_HMM_Component>::iterator min_itr1;
    vector<MSST_HMM_Component>::iterator min_itr2;

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

    MSST_HMM_Component hmmc1 = *min_itr1;
    MSST_HMM_Component hmmc2 = *min_itr2;

    MSST_HMM_Component hmmc_new;
    hmmc_new.left_child = new MSST_HMM_Component(hmmc1);
    hmmc_new.right_child = new MSST_HMM_Component(hmmc2);

    hmmc_new.weight = hmmc1.weight + hmmc2.weight;
    hmmc_new.div = minimum;


    vector<vector<StructureTensor> > samples1 = hmmc1.get_all_samples();
    vector<vector<StructureTensor> > samples2 = hmmc2.get_all_samples();

    samples1.insert(samples1.end(), samples2.begin(), samples2.end());

    hmmc_new.gauss.compute_from_samples(samples1);

    components.erase(min_itr2);
    components.erase(min_itr1);

    components.push_back(hmmc_new);

}

void MSST_HMM::addModel(const MSST_HMM &model)
{
    components.insert(components.end(), model.components.begin(), model.components.end());
}

void MSST_HMM::setModel(const Mat &model, const Mat &mask, const MSStructureTensorImage &img, int dim)
{
    MSST_MM<MSST_HMM_Component>::setModel(model);

    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == 1 )
            {
                int c = whichComponent( img.getTensor(p.x, p.y) );
                components[c].samples.push_back( img.getTensor(p.x, p.y) );
            }
        }
    }
}

void MSST_HMM::addModel( const Mat &model, const Mat &compIdxs, const Mat &mask, const MSStructureTensorImage &img, int dim) {
//    int modelsize = dim /* mean */ + dim * dim /* covariance */ + 1; /* weight */
    int nr_scales = (model.rows - 2) / 3;

    for(int i = 0; i< model.cols; i++)
    {
        int c = 0;
        MSST_Gaussian gauss;
        
        for(int s=0; s<nr_scales; s++)
        {
            StructureTensor st(model.at<double>(c,i), model.at<double>(c+1,i), model.at<double>(c+2,i));
            c += 3;
            gauss.mean.push_back(st);
        }
        gauss.cov = model.at<double>(c++,i);

        MSST_HMM_Component MMc;
        MMc.gauss = gauss;
        MMc.weight = model.at<double>(c++,i);

        Point p;
        for( p.y = 0; p.y < mask.rows; p.y++ )
        {
            for( p.x = 0; p.x < mask.cols; p.x++ )
            {
                if( mask.at<uchar>(p) == 1 && compIdxs.at<int>(p) == i)
                {
                    MMc.samples.push_back(img.getTensor(p.x, p.y));
                }
            }
        }
        components.push_back(MMc);

    }
}


void MSST_HMM::normalize_weights() {
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

MSST_HMM::~MSST_HMM() {
}

vector<vector<StructureTensor> > MSST_HMM_Component::get_all_samples() {
    vector<vector<StructureTensor> > result(500); //TODO: ist das noetig?

    if( right_child != NULL )
    {
        vector<vector<StructureTensor> > rsamples = right_child->get_all_samples();
        result.insert(result.end(), rsamples.begin(), rsamples.end());
    }
    if( left_child != NULL )
    {
        vector<vector<StructureTensor> > lsamples = left_child->get_all_samples();
        result.insert(result.end(), lsamples.begin(), lsamples.end());
    }

    result.insert(result.end(), samples.begin(), samples.end());
    return result;
}

/*MSST_HMM_Component::MSST_HMM_Component( Mat Component ) : MSST_GMM_Component(Component) {
    left_child = NULL;
    right_child = NULL;
    div = 0;
}*/

MSST_HMM_Component::MSST_HMM_Component(const MSST_HMM_Component& rhs) : MSST_GMM_Component(rhs) {
    left_child = NULL;
    right_child = NULL;
    if(rhs.left_child)
    {
        left_child = new MSST_HMM_Component(*rhs.left_child);
    }
    if(rhs.right_child)
    {
        right_child = new MSST_HMM_Component(*rhs.right_child);
    }
    samples = rhs.samples;
    div = rhs.div;
}

MSST_HMM_Component& MSST_HMM_Component::operator=(const MSST_HMM_Component& rhs) {
    
    if(this != &rhs)
    {
        MSST_HMM_Component* tmp_left = NULL;
        MSST_HMM_Component* tmp_right = NULL;
        if(rhs.left_child)
            tmp_left = new MSST_HMM_Component(*rhs.left_child);

        if(rhs.right_child)
            tmp_right = new MSST_HMM_Component(*rhs.right_child);

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

MSST_HMM_Component::MSST_HMM_Component() {
    left_child = NULL;
    right_child = NULL;
    div = 0;
}

MSST_HMM_Component::~MSST_HMM_Component() {
    delete right_child;
    delete left_child;
}
