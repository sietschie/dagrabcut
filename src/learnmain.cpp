#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <boost/program_options.hpp>

#include "hmm.hpp"
#include "grabcut.hpp"
#include "shared.hpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

void help()
{
    cout << "reads input_images and input_image_names.yml, generates\n"
         "HMM of the images for class_number\n"
         << endl;
}

void print_mean_variance(vector<double> list)
{
    double sum = 0;
    vector<double>::iterator itr;
    for(itr = list.begin(); itr != list.end(); itr++)
    {
        sum += *itr;
    }
    double mean = sum / list.size();

    double sum_squaredmeandiff = 0;

    for(itr = list.begin(); itr != list.end(); itr++)
    {
        double meandiff = *itr - mean;
        sum_squaredmeandiff += meandiff * meandiff;
    }
    double variance = sum_squaredmeandiff / list.size();

    cout << "Mean: " << mean << "   Variance: " << variance << endl;
}

void learnGMMfromSamples(vector<Vec3f> samples, Mat& model, int nr_gaussians = 5)
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

po::variables_map parseCommandline(int argc, char** argv)
{
    po::options_description generic("Generic options");
    generic.add_options()
        ("help,h", "produce help message")
        //("max-iterations,m", po::value<int>()->default_value(100), "maximum number of iterations")
        //("interactive,i", "interactive segmentation")
        ("gaussians,g", po::value<int>()->default_value(5), "number of gaussians used for the gmms")
        ("cluster,c", po::value<int>()->default_value(5), "number of gaussians used for the hmm")
    ;

    po::options_description hidden("Hidden options");
    hidden.add_options()
        ("class-number", po::value<int>()->required(), "the relevant class number")
        ("model", po::value<string>()->required(), "where to save the created model file")
        ("images", po::value< vector<string> >()->required(), "the input images")
    ;

    po::positional_options_description positional;
    positional.add("model", 1);
    positional.add("class-number", 1);
    positional.add("images", -1);

    po::options_description cmdline_options;
    cmdline_options.add(generic).add(hidden);

    po::options_description visible;
    visible.add(generic);

    po::variables_map vm;
 
    try {
        po::store(po::command_line_parser(argc, argv).options(cmdline_options).positional(positional).run(), vm);
        po::notify(vm);

    } catch ( std::exception& e )
    {
        cout << "Usage: ./learn.bin [options] modelfile class-number imagefile1 imagefile2 ...\n";
        cout << visible << endl;
        if(!vm.count("help"))
        {
            cout << e.what() << "\n";
            exit(0);
        }
        exit(1);
    }
    return vm;
}

int main( int argc, char** argv )
{
    po::variables_map vm = parseCommandline(argc, argv);

    vector< string > input_images = vm["images"].as< vector<string> >();
    int class_number = vm["class-number"].as<int>();
    string model_filename = vm["model"].as<string>();
    int nr_gaussians = vm["gaussians"].as<int>();

    HMM fgdHmm, bgdHmm;
    vector<HMM> fgdHmms, bgdHmms;

    for(vector<string>::iterator filename = input_images.begin(); filename != input_images.end(); ++filename)    
    {
        vector<Vec3f> bgdSamples, fgdSamples;
        HMM cur_fgdHmm, cur_bgdHmm;

        Mat image, mask;
        readImageAndMask(*filename, image, mask);

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

        Mat bgdModel, fgdModel;
        learnGMMfromSamples(bgdSamples, bgdModel);
        learnGMMfromSamples(fgdSamples, fgdModel);

        Mat binary_mask = mask.clone();
        for( p.y = 0; p.y < image.rows; p.y++ )
        {
            for( p.x = 0; p.x < image.cols; p.x++ )
            {
                binary_mask.at<uchar>(p) = mask.at<uchar>(p) == class_number ? 1 : 0;
            }
        }

        //Mat binary_mask = mask & 1;
        //Mat binary_mask = mask & class_number;
        cur_fgdHmm.add_model(fgdModel, binary_mask, image);
        cur_bgdHmm.add_model(bgdModel, 1 - binary_mask, image);

        //TODO: precompute the model, maybe cache it?

        /*string go_filename = imagename;
        go_filename.append(".grabcut-output.yml");

        FileStorage fs(go_filename, FileStorage::READ);
        Mat mask; fs["mask"] >> mask;
        Mat compIdxs; fs["componentIndexes"] >> compIdxs;
        Mat bgdModel; fs["bgdModel"] >> bgdModel;
            Mat fgdModel; fs["fgdModel"] >> fgdModel;

        cur_fgdHmm.add_model(fgdModel, compIdxs, mask & 1, image);
        cur_bgdHmm.add_model(bgdModel, compIdxs, 1 - (mask & 1), image);*/

        fgdHmms.push_back(cur_fgdHmm);
        bgdHmms.push_back(cur_bgdHmm);
    }

    vector<double> fgdDivs;
    for(int i=0; i<fgdHmms.size(); i++)
        for(int j=i+1; i<fgdHmms.size(); i++)
        {
            fgdDivs.push_back(fgdHmms[i].KLsym(fgdHmms[j]));
        }
    cout << "fgdDivs:  ";
    print_mean_variance(fgdDivs);

    vector<double> bgdDivs;
    for(int i=0; i<bgdHmms.size(); i++)
        for(int j=i+1; i<bgdHmms.size(); i++)
        {
            bgdDivs.push_back(bgdHmms[i].KLsym(bgdHmms[j]));
        }
    cout << "bgdDivs:  ";
    print_mean_variance(bgdDivs);

    vector<double> betweenDivs;
    for(int i=0; i<bgdHmms.size(); i++)
        for(int j=0; i<fgdHmms.size(); i++)
        {
            betweenDivs.push_back(bgdHmms[i].KLsym(bgdHmms[j]));
        }
    cout << "betweenDivs:  ";
    print_mean_variance(betweenDivs);


    for(int i=0; i<fgdHmms.size(); i++)
        fgdHmm.add_model(fgdHmms[i]);
    for(int i=0; i<bgdHmms.size(); i++)
        bgdHmm.add_model(bgdHmms[i]);

    cout << "KLdiv: " << fgdHmm.KLdiv(bgdHmm) << endl;
    cout << "KLdiv: " << bgdHmm.KLdiv(fgdHmm) << endl;

    fgdHmm.normalize_weights();
    bgdHmm.normalize_weights();

    while( fgdHmm.getComponentsCount() > vm["cluster"].as<int>())
    {
        fgdHmm.cluster_once();
        bgdHmm.cluster_once();
    }

    FileStorage fs2(model_filename, FileStorage::WRITE);
    fs2 << "files" << "[";
    for(vector<string>::iterator filename = input_images.begin(); filename != input_images.end(); ++filename)    
    {
        fs2 << *filename;
    }
    fs2 << "]";
    fs2 << "fgdHmm" << fgdHmm;
    fs2 << "bgdHmm" << bgdHmm;
    fs2.release();

    return 0;
}
