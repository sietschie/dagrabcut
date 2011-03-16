#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "grabcut.hpp"
#include "shared.hpp"

#include <iostream>
#include <boost/program_options.hpp>

using namespace std;
using namespace cv;
namespace po = boost::program_options;

void help()
{
    cout << "reads input_images and input_image_names.yml, generates\n"
         "GMM of the combined images for class_number\n"
         << endl;
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

    GMM gmm(model, nr_gaussians);

    gmm.initLearning();
    for( int i = 0; i < (int)samples.size(); i++ )
        gmm.addSample( labels.at<int>(i,0), samples[i] );
    gmm.endLearning();
}

po::variables_map parseCommandline(int argc, char** argv)
{
    po::options_description generic("Generic options");
    generic.add_options()
        ("help,h", "produce help message")
        //("max-iterations,m", po::value<int>()->default_value(100), "maximum number of iterations")
        //("interactive,i", "interactive segmentation")
        ("gaussians,g", po::value<int>()->default_value(5), "number of gaussians used for the gmms")
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

    vector<Vec3f> bgdSamples, fgdSamples;

    for(vector<string>::iterator filename = input_images.begin(); filename != input_images.end(); ++filename)    
    {
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

    }

    Mat bgdModel, fgdModel;
    learnGMMfromSamples(bgdSamples, bgdModel);
    learnGMMfromSamples(fgdSamples, fgdModel);

    FileStorage fs2(model_filename, FileStorage::WRITE);
    fs2 << "fgdModel" << fgdModel;
    fs2 << "bgdModel" << bgdModel;

    return 0;
}
