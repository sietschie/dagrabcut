#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "../lib/grabcut.hpp"
#include "../lib/hmm.hpp"
#include "../lib/shared.hpp"

#include <iostream>
#include <boost/program_options.hpp>

using namespace std;
using namespace cv;
namespace po = boost::program_options;

void help()
{
    cout << "reads model and image, generates Segmentation for class_number\n"
         << endl;
}

void getBinMask( const Mat& comMask, Mat& binMask )
{
    if( comMask.empty() || comMask.type()!=CV_8UC1 )
        CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

class GCApplication
{
public:
    void setImageAndWinName( const Mat& _image, const string& _winName, Mat& _bgdModel, Mat& _fgdModel );
    void showImage() const;
    int nextIter(int max_iterations);
    int getIterCount() const {
        return iterCount;
    }
    Mat mask;
    Mat initial_mask;
private:
    const string* winName;
    const Mat* image;
    Mat input_mask;
    Mat bgdModel, fgdModel;

    bool isInitialized;

    int iterCount;
};

void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName, Mat& _bgdModel, Mat& _fgdModel )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;

    bgdModel = _bgdModel;
    fgdModel = _fgdModel;
}

void GCApplication::showImage() const
{
    if( image->empty() || winName->empty() )
        return;

    Mat res;
    Mat binMask;
    if( !isInitialized )
        image->copyTo( res );
    else
    {
        getBinMask( mask, binMask );

        image->copyTo( res, binMask );
    }

    imshow( *winName, res );
}

int GCApplication::nextIter(int max_iterations = 2)
{
    isInitialized = true;
    Rect rect;

    cg_grabCut( *image, mask, initial_mask, rect, bgdModel, fgdModel, max_iterations );

    iterCount += max_iterations;

    return iterCount;
}

GCApplication gcapp;


void compareMasks(const Mat &gt,const Mat& segm, int class_number, int& true_positive, int& true_negative, int& false_positive, int& false_negative, int& unknown)
{
    true_positive = 0;
    true_negative = 0;
    false_positive = 0;
    false_negative = 0;
    unknown = 0;

    assert(gt.rows == segm.rows && gt.cols == segm.cols);

    Point p;
    for( p.y = 0; p.y < gt.rows; p.y++ )
    {
        for( p.x = 0; p.x < gt.cols; p.x++ )
        {
            if(gt.at<uchar>(p) == class_number)
            {
                if(segm.at<uchar>(p) == GC_FGD | segm.at<uchar>(p) == GC_PR_FGD)
                {
                    true_positive++;
                } else {
                    false_negative++;
                }
            } else if (gt.at<uchar>(p) == 255) {
                unknown++;
            } else {
                if(segm.at<uchar>(p) == GC_FGD | segm.at<uchar>(p) == GC_PR_FGD)
                {
                    false_positive++;
                } else {
                    true_negative++;
                }
            }
        }
    }
    assert(gt.rows * gt.cols == true_positive + true_negative + false_positive + false_negative + unknown);

}

po::variables_map parseCommandline(int argc, char** argv)
{
    po::options_description generic("Generic options");
    generic.add_options()
        ("help,h", "produce help message")
        ("max-iterations,m", po::value<int>()->default_value(100), "maximum number of iterations")
        ("interactive,i", "interactive segmentation")
        //("gaussians,g", po::value<int>()->default_value(5), "number of gaussians used for the gmms")
    ;

    po::options_description hidden("Hidden options");
    hidden.add_options()
        ("class-number", po::value<int>()->required(), "the relevant class number")
        ("model", po::value<string>()->required(), "where to read the model from")
        ("image", po::value<string>()->required(), "the input image")
        ("output", po::value<string>()->required(), "the output file")
    ;

    po::positional_options_description positional;
    positional.add("model", 1);
    positional.add("class-number", 1);
    positional.add("image", 1);
    positional.add("output", 1);

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
        cout << "Usage: ./test.bin [options] modelfile class-number imagefile outputfile\n";
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

    string input_image = vm["image"].as<string>();
    int class_number = vm["class-number"].as<int>();
    string model_filename = vm["model"].as<string>();
    //int nr_gaussians = vm["gaussians"].as<int>();
    bool interactive = vm.count("interactive");
    int max_iterations = vm["max-iterations"].as<int>();
    string output_filename = vm["output"].as<string>();

    vector<Vec3f> bgdSamples, fgdSamples;

    HMM fgdHmm, bgdHmm;
    FileStorage fs(model_filename, FileStorage::READ);
    readHMM(fs["fgdHmm"], fgdHmm);
    readHMM(fs["bgdHmm"], bgdHmm);

    double var_bgd_kl_sym, var_bgd_kl_mr, var_bgd_kl_rm, var_fgd_kl_sym, var_fgd_kl_mr, var_fgd_kl_rm;
    fs["var_bgd_kl_sym"] >> var_bgd_kl_sym;
    fs["var_bgd_kl_mr"] >> var_bgd_kl_mr;
    fs["var_bgd_kl_rm"] >> var_bgd_kl_rm;
    fs["var_fgd_kl_sym"] >> var_fgd_kl_sym;
    fs["var_fgd_kl_mr"] >> var_fgd_kl_mr;
    fs["var_fgd_kl_rm"] >> var_fgd_kl_rm;

    fs.release();

    Mat image, mask;
    readImageAndMask(input_image, image, mask);

    const string winName = "image";
    cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );

    Mat bgdModel = bgdHmm.getModel();
    Mat fgdModel = fgdHmm.getModel();
    Mat fgdModel_cloned = fgdModel.clone();

    gcapp.setImageAndWinName( image, winName, bgdModel, fgdModel );
    gcapp.showImage();

    if(interactive)
        cvWaitKey(1000);

    int iterCount = gcapp.getIterCount();
    cout << "<" << iterCount << "... ";
    int newIterCount = gcapp.nextIter(max_iterations);
    if( newIterCount > iterCount )
    {
        gcapp.showImage();
        cout << newIterCount << ">" << endl;
    }

    FileStorage fs2(output_filename, FileStorage::WRITE);
    fs2 << "input_image" << input_image;
    fs2 << "mask" << gcapp.mask;
    fs2 << "initial_mask" << gcapp.initial_mask;
    fs2 << "fgdModel" << fgdModel;
    fs2 << "bgdModel" << bgdModel;

    int tp, tn, fp, fn, unknown;
    compareMasks(mask, gcapp.mask, class_number, tp, tn, fp, fn, unknown);
    cout << "true positive: " << tp << ", true negative: " << tn << ", false positive: " << fp << ", false negative: " << fn << ", unknown: " << unknown << endl;

    double fgd_rate = tp / (double) (tp + fn);
    double bgd_rate = tn / (double) (tn + fp);
    double joint_rate = (fgd_rate + bgd_rate) / 2;

    cout << "fgd: " << fgd_rate << ", bgd: " << bgd_rate << ", joint: " << joint_rate;

    fs2 << "true positive" << tp;
    fs2 << "true negative" << tn;
    fs2 << "false positive" << fp;
    fs2 << "false negative" << fn;

    fs2 << "fgd" << fgd_rate;
    fs2 << "bgd" << bgd_rate;
    fs2 << "joint" << joint_rate;

    {
        Mat model = fgdHmm.getModel(); 
        GMM i; i.setModel(model);
        GMM r; r.setModel(fgdModel);
        double kl_div_i_r = i.KLdiv(r);
        double kl_div_r_i = r.KLdiv(i);
        double kl_sym = i.KLsym(r);

        cout << " ,fgd KL input result: " << kl_div_i_r;
        cout << " ,prob fgd KL input result: " << compute_probability( kl_div_i_r, var_fgd_kl_mr);
        cout << " ,fgd KL result input: " << kl_div_r_i;
        cout << " ,prob fgd KL result input: " << compute_probability( kl_div_r_i, var_fgd_kl_rm);
        cout << " ,fgd KL sym: " << kl_sym;
        cout << " ,prob fgd KL sym: " << compute_probability( kl_sym, var_fgd_kl_sym);
        fs2 << "prob fgd KL sym" << compute_probability( kl_sym, var_fgd_kl_sym);
    }


    {
        Mat model = bgdHmm.getModel(); 
        GMM i; i.setModel(model);
        GMM r; r.setModel(bgdModel);
        double kl_div_i_r = i.KLdiv(r);
        double kl_div_r_i = r.KLdiv(i);
        double kl_sym = i.KLsym(r);

        cout << " ,bgd KL input result: " << kl_div_i_r;
        cout << " ,prob bgd KL input result: " << compute_probability( kl_div_i_r, var_bgd_kl_mr);
        cout << " ,bgd KL result input: " << kl_div_r_i;
        cout << " ,prob bgd KL result input: " << compute_probability( kl_div_r_i, var_bgd_kl_rm);
        cout << " ,bgd KL sym: " << kl_sym;
        cout << " ,prob bgd KL sym: " << compute_probability( kl_sym, var_bgd_kl_sym);
    }
    cout << endl;
    if(interactive)
        cvWaitKey(0);

exit_main:
    cvDestroyWindow( winName.c_str() );
    return 0;
}
