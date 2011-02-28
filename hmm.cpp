#include "hmm.hpp"
#include <limits>
#include <iostream>
#include <algorithm>
#include <vector>

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

cv::FileStorage& operator<<(cv::FileStorage& fs, const HMM_Component& component)
{
    fs << "{";
    fs << "weight" << component.weight;
    fs << "gauss" << component.gauss;

    std::vector<cv::Vec3b>::const_iterator itr;
    int size = component.samples.size();
    fs << "nr_samples" << size;
    fs << "samples" << "[";
    for(itr = component.samples.begin(); itr != component.samples.end(); itr++)
    {
        fs << *itr;
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

void readGaussian(const cv::FileNode& fn, Gaussian& gauss)
{
    fn["mean"] >> gauss.mean;
    fn["cov"] >> gauss.cov;
}

void readHMM_Component(const cv::FileNode& fn, HMM_Component& hmmc)
{
    double w = (double)fn["weight"];
    hmmc.weight = w;
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
void HMM::add_model(Mat model, Mat compIdxs, Mat mask, Mat img, int dim) {
    int modelsize = dim /* mean */ + dim * dim /* covariance */ + 1; /* weight */

    int componentsCount = model.cols / modelsize;

    std::cout << "componentsCount: " << componentsCount << std::endl;

    double *coefs = model.ptr<double>(0);
    double *mean = coefs + componentsCount;
    double *cov = mean + dim*componentsCount;

    HMM_Component *hmmc;

    for(int i = 0; i<componentsCount;i++)
    {
        hmmc = new HMM_Component();
        hmmc->weight = coefs[i];
        
        double *cpy_mean = new double[dim];
        std::copy(mean + (i*dim), mean + (i*dim) + dim, cpy_mean);
        hmmc->gauss.mean = Mat(dim, 1, CV_64F, cpy_mean);

        double *cpy_cov = new double[dim * dim];
        std::copy(cov + (i*dim*dim), cov + (i*dim*dim) + dim*dim, cpy_cov);
        hmmc->gauss.cov = Mat(dim,dim, CV_64F, cpy_cov);

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


HMM::~HMM() {
    vector<HMM_Component*>::iterator itr;

    for(itr=components.begin();itr!=components.end();itr++)
    {
        delete *itr;
    }
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
}

HMM_Component::~HMM_Component() {
    delete right_child;
    delete left_child;
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
    double summand2 = trace( cov1_inv * g2.cov + cov * g2.cov )[0];

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

    for(itr=samples.begin(); itr!=samples.end();itr++)
    {
        mean.at<double>(0,0) += (*itr)[0];
        mean.at<double>(1,0) += (*itr)[1];
        mean.at<double>(2,0) += (*itr)[2];

        for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            cov.at<double>(i,j) += (*itr)[i] * (*itr)[j];
    }

    mean = mean / samples.size();
    cov = cov / samples.size();

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
