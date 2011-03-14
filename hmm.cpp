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
    for(int i = 0; i<size;i++)
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

void readHMM(const cv::FileNode& fn, HMM& hmm){
    int size =(int)fn["nr_components"];
    //std::cout << size << std::endl;

    FileNodeIterator fni = fn["components"].begin();
    for(int i = 0; i<size;i++)
    {
        HMM_Component *hmmc = new HMM_Component();
        readHMM_Component((*fni), *hmmc);
        fni++;
        hmm.components.push_back(hmmc);
    }
}

cv::FileStorage& operator<<(cv::FileStorage& fs, const HMM& hmm)
{
    vector<HMM_Component*>::const_iterator itr;

    std::vector<HMM_Component*> components;

//    fs << "[";

    fs << "{";// << "CV_PI" << "1+1" << "month" << 12 << "day" << 31 << "year" << 1969 << "g" << "}";
    int size = hmm.components.size();
    fs << "nr_components" << size;
    fs << "components" << "[";
    for(itr = (hmm.components).begin(); itr != hmm.components.end(); itr++)
    {
        fs << **itr;
    }
    //std::cout << "hallo" << endl;
    fs << "]" << "}";
    return fs;
}


void HMM::cluster_once()
{
    vector<HMM_Component*>::iterator itr1;
    vector<HMM_Component*>::iterator itr2;

    vector<HMM_Component*>::iterator min_itr1;
    vector<HMM_Component*>::iterator min_itr2;

    double minimum = 1e+100;

    for(itr1 = components.begin(); itr1 != components.end(); itr1++)
    for(itr2 = (itr1 + 1); itr2 != components.end(); itr2++)
    {
        double div = (*itr1)->gauss.KLsym((*itr2)->gauss);
        if(div < minimum) {
            min_itr1 = itr1;
            min_itr2 = itr2;
            minimum = div;
        }
    }
    
    HMM_Component *hmmc1 = *min_itr1;
    HMM_Component *hmmc2 = *min_itr2;

    components.erase(min_itr2);    
    components.erase(min_itr1);    

    HMM_Component *hmmc_new = new HMM_Component;

    hmmc_new->weight = hmmc1->weight + hmmc2->weight;
    hmmc_new->div = minimum;

    hmmc_new->left_child = hmmc1;
    hmmc_new->right_child = hmmc2;

    vector<cv::Vec3b> samples1 = hmmc1->get_all_samples();
    vector<cv::Vec3b> samples2 = hmmc2->get_all_samples();

    samples1.insert(samples1.end(), samples2.begin(), samples2.end());

    hmmc_new->gauss.compute_from_samples(samples1);

    components.push_back(hmmc_new);

/*    std::vector< cv::Vec3b > all_samples( samples1.size() + samples2.size() );
    copy(samples1.begin(), samples1.end(), all_samples.end());    
    copy(samples2.begin(), samples2.end(), all_samples.end());    

    std::cout << minimum << std::endl;
*/

}

void HMM::add_model(HMM &model)
{
    components.insert(components.end(), model.components.begin(), model.components.end());
}

void HMM::add_model(Mat &model, const Mat &mask, const Mat &img, int dim)
{
    int modelsize = dim /* mean */ + dim * dim /* covariance */ + 1; /* weight */

    int componentsCount = model.cols;

    //std::cout << "componentsCount: " << componentsCount << std::endl;

    double *coefs = model.ptr<double>(0);
    double *mean = coefs + componentsCount;
    double *cov = mean + dim*componentsCount;

    HMM_Component *hmmc;

    for(int i = 0; i<componentsCount;i++)
    {
        Gaussian gauss;
        int c=0;
        gauss.mean = Mat(3,1, CV_64FC1);
        for(int j=0; j < 3; j++)
        {
            double value = model.at<double>(c,i);
            gauss.mean.at<double>(j,0) = value;
            c++;
        }

        gauss.cov = Mat(3,3, CV_64FC1);
        for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)
        {
            gauss.cov.at<double>(j,k) = model.at<double>(c,i);
            c++;
        }       

        hmmc = new HMM_Component();
        hmmc->gauss = gauss;
        hmmc->weight = model.at<double>(c,i);

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
                components[c]->samples.push_back(img.at<Vec3b>(p));
            }
        }
    }
}

int HMM::whichComponent( const Vec3d color ) const
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < components.size(); ci++ )
    {
		double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}



void HMM::add_model( Mat &model, const Mat &compIdxs, const Mat &mask, const Mat &img, int dim) {
    int modelsize = dim /* mean */ + dim * dim /* covariance */ + 1; /* weight */

    int componentsCount = model.cols / modelsize;

    //std::cout << "componentsCount: " << componentsCount << std::endl;

    double *coefs = model.ptr<double>(0);
    double *mean = coefs + componentsCount;
    double *cov = mean + dim*componentsCount;

    HMM_Component *hmmc;

    for(int i = 0; i<componentsCount;i++)
    {
        hmmc = new HMM_Component();
        hmmc->weight = coefs[i];
        
        double *cpy_mean = mean + (i*dim);
        //std::copy(mean + (i*dim), mean + (i*dim) + dim, cpy_mean);
		hmmc->gauss.mean = Mat(dim, 1, CV_64F);
		for(int j=0;j<dim;j++)
		{
			hmmc->gauss.mean.at<double>(j,0) = cpy_mean[j];
		}
        //hmmc.gauss.mean = Mat(dim, 1, CV_64F, cpy_mean);

        double *cpy_cov = cov + (i*dim*dim);
        //std::copy(cov + (i*dim*dim), cov + (i*dim*dim) + dim*dim, cpy_cov);
        hmmc->gauss.cov = Mat(dim,dim, CV_64F);
		for(int j=0;j<dim;j++)
		for(int k=0;k<dim;k++)
		{
			hmmc->gauss.cov.at<double>(j,k) = cpy_cov[j + k * dim];
		}

        //TODO: muss der speicher wieder freigegeben werden?

        Point p;
        for( p.y = 0; p.y < mask.rows; p.y++ )
        {
            for( p.x = 0; p.x < mask.cols; p.x++ )
            {
                if( mask.at<uchar>(p) == 1 && compIdxs.at<int>(p) == i)
                {
                    hmmc->samples.push_back(img.at<Vec3b>(p));
                }
            }
        }
        components.push_back(hmmc);

    }
}

cv::Mat HMM::get_model() {
    int dim = 3;
    int modelSize = (dim + 1) * dim + 1;
    Mat model = Mat( modelSize, components.size(), CV_64FC1 );
    for(int i=0;i<components.size();i++)
    {
        int counter = 0;
        for(int j=0;j<dim; j++)
        {
            model.at<double>( counter, i) = components[i]->gauss.mean.at<double>(j,0);
            counter++;
        }

        for(int j=0;j<dim; j++)
        for(int k=0;k<dim; k++)
        {
            model.at<double>(counter, i) = components[i]->gauss.cov.at<double>(j,k);
            counter++;
        }
        model.at<double>(counter, i) = components[i]->weight;
    }
    return model;
}

double HMM::KLsym(HMM& rhs)
{
    return KLdiv(rhs) + rhs.KLdiv(*this);
}

double HMM::KLdiv(const HMM& rhs)
{
    vector<int> mapping(components.size());
    for(int i=0;i<components.size();i++)
    {
        int min_component = 0;
        double min = 10e+100; //TODO: richtig grossen wert aus limits nehmen
        for(int j=0;j<rhs.components.size();j++)
        {
            double div = components[i]->gauss.KLdiv(rhs.components[j]->gauss) - log(rhs.components[j]->weight);
            if( div < min )
            {
                min = div;
                min_component = j;
            }
            mapping[i] = min_component;
        }
    }
    
    double div = 0.0;
    for(int i=0;i<components.size();i++)
    {
        double kl = components[i]->gauss.KLdiv(rhs.components[mapping[i]]->gauss);
        double summand =  log( components[i]->weight / rhs.components[mapping[i]]->weight);
        div += components[i]->weight * (kl + summand);
    }
    return div;
}

void HMM::normalize_weights() {
    double sum = 0.0;
    for(int i=0;i<components.size();i++)
    {
        sum += components[i]->weight;
    }

    for(int i=0;i<components.size();i++)
    {
        components[i]->weight /= sum;
    }
}

double HMM::operator()( const Vec3d color ) const
{
    double res = 0;
    for( int ci = 0; ci < components.size(); ci++ )
        res += components[ci]->weight * (*this)(ci, color );
    return res;
}

double HMM::operator()( int ci, const Vec3d color ) const
{
    double res = 0;
    HMM_Component* hmmc = components[ci];
    if( hmmc->weight > 0 )
    {
        double covDeterms = determinant(hmmc->gauss.cov);
		CV_Assert( covDeterms > std::numeric_limits<double>::epsilon() );
        Vec3d diff = color;
        diff[0] -= hmmc->gauss.mean.at<double>(0,0);
        diff[1] -= hmmc->gauss.mean.at<double>(1,0);
        diff[2] -= hmmc->gauss.mean.at<double>(2,0);

        Mat inverseCovs = hmmc->gauss.cov.inv();

        double mult = 0.0;
        for(int i=0;i<3;i++)
        {
            double row = 0.0;
            for(int j=0;j<3;j++)
            {
                row += diff[j] * inverseCovs.at<double>(j,i);
            }
            mult += diff[i] * row;
        }

        //double test = diff * (inverseCovs * diff);

        //double mult = diff[0]*(diff[0]*inverseCovs[0][0] + diff[1]*inverseCovs[1][0] + diff[2]*inverseCovs[2][0])
        //           + diff[1]*(diff[0]*inverseCovs[0][1] + diff[1]*inverseCovs[1][1] + diff[2]*inverseCovs[2][1])
        //           + diff[2]*(diff[0]*inverseCovs[0][2] + diff[1]*inverseCovs[1][2] + diff[2]*inverseCovs[2][2]);
        res = 1.0f/sqrt(covDeterms) * exp(-0.5f*mult);
    }
    return res;
}


void HMM::free_components()
{
    vector<HMM_Component*>::iterator itr;

    for(itr=components.begin();itr!=components.end();itr++)
    {
        delete *itr;
    }
}

HMM::~HMM() {
}

vector<cv::Vec3b> HMM_Component::get_all_samples(){
    vector<cv::Vec3b> result(500);

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

HMM_Component::HMM_Component() {
    left_child = NULL;
    right_child = NULL;
    div = 0;
}

HMM_Component::~HMM_Component() {
    delete right_child;
    delete left_child;
}
