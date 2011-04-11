#include "shared.hpp"

#include <iostream>
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "gmm.hpp"
#include "msst_gmm.hpp"

using namespace std;
using namespace cv;

void readImageAndMask(string filename, Mat& image, Mat& mask)
{
    cout << "Reading " << filename << "..." << endl;

    if( filename.empty() )
    {
        cout << "\nDurn, couldn't read in " << filename << endl;
        return;
    }
    image = imread( filename, 1 );
    if( image.empty() )
    {
        cout << "\n Durn, couldn't read image filename " << filename << endl;
        return;
    }

    string mask_filename = filename;
    mask_filename.append(".yml");

    FileStorage fs(mask_filename, FileStorage::READ);
    fs["mask"] >> mask;
}

void compute_variance(vector<string> input_images, Mat mean_bgdModel, Mat mean_fgdModel, int nr_gaussians, int class_number, string model_filename,
        double &var_bgd_kl_sym, double &var_bgd_kl_mr, double &var_bgd_kl_rm, double &var_fgd_kl_sym, double &var_fgd_kl_mr, double &var_fgd_kl_rm )
{
    GMM mean_bgdGMM, mean_fgdGMM;
    mean_bgdGMM.setModel(mean_bgdModel);
    mean_fgdGMM.setModel(mean_fgdModel);

    vector<double> diff_bgd_kl_sym;
    vector<double> diff_bgd_kl_mr;
    vector<double> diff_bgd_kl_rm;
    vector<double> diff_fgd_kl_sym;
    vector<double> diff_fgd_kl_mr;
    vector<double> diff_fgd_kl_rm;

    for(vector<string>::iterator filename = input_images.begin(); filename != input_images.end(); ++filename)    
    {
        vector<Vec3f> bgdSamples, fgdSamples;

        Mat image, mask;
        readImageAndMask(*filename, image, mask);

        Mat bgdModel, fgdModel;
        computeGMM(*filename, image, mask, model_filename, class_number, bgdModel, fgdModel);

        GMM bgdGMM, fgdGMM;
        bgdGMM.setModel(bgdModel);
        fgdGMM.setModel(fgdModel);

        diff_bgd_kl_sym.push_back( mean_bgdGMM.KLsym(bgdGMM) );        
        diff_bgd_kl_mr.push_back( mean_bgdGMM.KLdiv(bgdGMM) );        
        diff_bgd_kl_rm.push_back( bgdGMM.KLdiv(mean_bgdGMM) );        

        diff_fgd_kl_sym.push_back( mean_fgdGMM.KLsym(fgdGMM) );        
        diff_fgd_kl_mr.push_back( mean_fgdGMM.KLdiv(fgdGMM) );        
        diff_fgd_kl_rm.push_back( fgdGMM.KLdiv(mean_fgdGMM) );        

    }

    var_fgd_kl_sym = compute_variance_from_vector( diff_bgd_kl_sym );
    var_fgd_kl_mr = compute_variance_from_vector( diff_bgd_kl_mr );
    var_fgd_kl_rm = compute_variance_from_vector( diff_bgd_kl_rm );
    var_bgd_kl_sym = compute_variance_from_vector( diff_fgd_kl_sym );
    var_bgd_kl_mr = compute_variance_from_vector( diff_fgd_kl_mr );
    var_bgd_kl_rm = compute_variance_from_vector( diff_fgd_kl_rm );
    
}

void MSST_compute_variance(vector<string> input_images, Mat mean_bgdModel, Mat mean_fgdModel, int nr_gaussians, int class_number, string model_filename,
        double &var_bgd_kl_sym, double &var_bgd_kl_mr, double &var_bgd_kl_rm, double &var_fgd_kl_sym, double &var_fgd_kl_mr, double &var_fgd_kl_rm )
{
    MSST_GMM mean_bgdGMM, mean_fgdGMM;
    mean_bgdGMM.setModel(mean_bgdModel);
    mean_fgdGMM.setModel(mean_fgdModel);

    vector<double> diff_bgd_kl_sym;
    vector<double> diff_bgd_kl_mr;
    vector<double> diff_bgd_kl_rm;
    vector<double> diff_fgd_kl_sym;
    vector<double> diff_fgd_kl_mr;
    vector<double> diff_fgd_kl_rm;

    for(vector<string>::iterator filename = input_images.begin(); filename != input_images.end(); ++filename)    
    {
        vector<vector<StructureTensor> > bgdSamples, fgdSamples;

        Mat image, mask;
        readImageAndMask(*filename, image, mask);

        MSStructureTensorImage stimage(image);

        Mat bgdModel, fgdModel;
        MSST_computeGMM(*filename, stimage, mask, model_filename, class_number, bgdModel, fgdModel);

        MSST_GMM bgdGMM, fgdGMM;
        bgdGMM.setModel(bgdModel);
        fgdGMM.setModel(fgdModel);

        diff_bgd_kl_sym.push_back( mean_bgdGMM.KLsym(bgdGMM) );        
        diff_bgd_kl_mr.push_back( mean_bgdGMM.KLdiv(bgdGMM) );        
        diff_bgd_kl_rm.push_back( bgdGMM.KLdiv(mean_bgdGMM) );        

        diff_fgd_kl_sym.push_back( mean_fgdGMM.KLsym(fgdGMM) );        
        diff_fgd_kl_mr.push_back( mean_fgdGMM.KLdiv(fgdGMM) );        
        diff_fgd_kl_rm.push_back( fgdGMM.KLdiv(mean_fgdGMM) );        

    }

    var_fgd_kl_sym = compute_variance_from_vector( diff_bgd_kl_sym );
    var_fgd_kl_mr = compute_variance_from_vector( diff_bgd_kl_mr );
    var_fgd_kl_rm = compute_variance_from_vector( diff_bgd_kl_rm );
    var_bgd_kl_sym = compute_variance_from_vector( diff_fgd_kl_sym );
    var_bgd_kl_mr = compute_variance_from_vector( diff_fgd_kl_mr );
    var_bgd_kl_rm = compute_variance_from_vector( diff_fgd_kl_rm );
}

void MSST_learnGMMfromSamples(const vector<vector<StructureTensor> > &samples, Mat& model, int nr_gaussians)
{
    const int kMeansItCount = 5;
    const int kMeansType = KMEANS_PP_CENTERS;
    const int componentsCount = nr_gaussians;

    Mat labels;
    assert(samples.size() != 0);

    vector<vector<StructureTensor> > tmp_centers;

    MSST_kmeans( samples, componentsCount,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 1, labels, tmp_centers);

    cout << "start learning GMM..." << endl;

    MSST_GMM gmm;
    gmm.setComponentsCount(nr_gaussians);

    gmm.initLearning();
    for( int i = 0; i < (int)samples.size(); i++ )
        gmm.addSample( labels.at<int>(i,0), samples[i] );
    gmm.endLearning();
    model = gmm.getModel();

    assert(model.cols == nr_gaussians);
}


void learnGMMfromSamples(vector<Vec3f> samples, Mat& model, int nr_gaussians)
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;
    const int componentsCount = nr_gaussians;

    Mat labels;
    CV_Assert( !samples.empty() );
    Mat _samples( (int)samples.size(), 3, CV_32FC1, &samples[0][0] );

    kmeans( _samples, componentsCount, labels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType, 0 );

    cout << "start learning GMM..." << endl;

    GMM gmm;
    gmm.setComponentsCount(nr_gaussians);

    gmm.initLearning();
    for( int i = 0; i < (int)samples.size(); i++ )
        gmm.addSample( labels.at<int>(i,0), samples[i] );
    gmm.endLearning();
    model = gmm.getModel();

    assert(model.cols == nr_gaussians);
}

double compute_variance_from_vector(vector<double> diffs)
{
    double square_sum = 0.0;
    for(vector<double>::iterator itr = diffs.begin(); itr != diffs.end(); itr++)
    {
        square_sum += *itr * *itr;
    }
    
    double variance = square_sum / diffs.size();
    return variance;
}

double compute_probability(double dist, double variance)
{
    double exponent = (dist * dist) / (-2 * variance);
    double res = exp( exponent ) / sqrt( 2.0 * M_PI * variance );

    return res;
}

void MSST_computeGMM(std::string filename, const MSStructureTensorImage& stimage, const cv::Mat& mask, std::string model_filename, int class_number, cv::Mat& bgdModel, cv::Mat& fgdModel)
{
    size_t pos = model_filename.find_last_of('/');
    string model_path = model_filename.substr(0,pos+1);


    pos = filename.find_last_of('/');
    string image_basename = filename.substr(pos+1);

    string gmm_filename = model_path + image_basename + ".msst_gmm.yml";

    cout << "model_path = " << model_path;
    cout << "  image_basename = " << image_basename;
    cout << "  gmm_filename = " << gmm_filename << endl;

//    ifstream ifs(gmm_filename, ifstream::in);
    ifstream ifs(gmm_filename.c_str(), ifstream::in);
    if(ifs.good())
    {
        cout << "DATEI EXISTIERT BEREITS" << endl;
        FileStorage fs(gmm_filename, FileStorage::READ);
        fs["bgdModel"] >> bgdModel;
        fs["fgdModel"] >> fgdModel;
        fs.release();
    }
    else {
        vector<vector<StructureTensor> > bgdSamples, fgdSamples;

        Point p;
        for( p.y = 0; p.y < stimage.rows; p.y++ )
        {
            for( p.x = 0; p.x < stimage.cols; p.x++ )
            {
                if( mask.at<uchar>(p) != class_number)
                    bgdSamples.push_back( stimage.getTensor(p.x, p.y) );
                else // GC_FGD | GC_PR_FGD
                    fgdSamples.push_back( stimage.getTensor(p.x, p.y) );
            }
        }

        MSST_learnGMMfromSamples(bgdSamples, bgdModel);
        MSST_learnGMMfromSamples(fgdSamples, fgdModel);

        FileStorage fs(gmm_filename, FileStorage::WRITE);
        fs << "bgdModel" << bgdModel;
        fs << "fgdModel" << fgdModel;
        fs.release();

    }
}

void computeGMM(std::string filename, const cv::Mat& image, const cv::Mat& mask, std::string model_filename, int class_number, cv::Mat& bgdModel, cv::Mat& fgdModel)
{

    size_t pos = model_filename.find_last_of('/');
    string model_path = model_filename.substr(0,pos+1);


    pos = filename.find_last_of('/');
    string image_basename = filename.substr(pos+1);

    string gmm_filename = model_path + image_basename + ".gmm.yml";

    cout << "model_path = " << model_path;
    cout << "  image_basename = " << image_basename;
    cout << "  gmm_filename = " << gmm_filename << endl;

//    ifstream ifs(gmm_filename, ifstream::in);
    ifstream ifs(gmm_filename.c_str(), ifstream::in);
    if(ifs.good())
    {
        cout << "DATEI EXISTIERT BEREITS" << endl;
        FileStorage fs(gmm_filename, FileStorage::READ);
        fs["bgdModel"] >> bgdModel;
        fs["fgdModel"] >> fgdModel;
        fs.release();
    }
    else {
        vector<Vec3f> bgdSamples, fgdSamples;

        Point p;
        for( p.y = 0; p.y < image.rows; p.y++ )
        {
            for( p.x = 0; p.x < image.cols; p.x++ )
            {
                if( mask.at<uchar>(p) != class_number)
                    bgdSamples.push_back( (Vec3f)image.at<Vec3b>(p) );
                else // GC_FGD | GC_PR_FGD
                    fgdSamples.push_back( (Vec3f)image.at<Vec3b>(p) );
            }
        }

        learnGMMfromSamples(bgdSamples, bgdModel);
        learnGMMfromSamples(fgdSamples, fgdModel);

        FileStorage fs(gmm_filename, FileStorage::WRITE);
        fs << "bgdModel" << bgdModel;
        fs << "fgdModel" << fgdModel;
        fs.release();

    }
}


