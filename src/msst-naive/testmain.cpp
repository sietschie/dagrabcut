#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "../lib/structuretensor.hpp"
#include "../lib/gmm.hpp"
#include "../lib/cmsst_grabcut.hpp"
#include "../lib/shared.hpp"

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
    void setImageAndWinName( const Mat& _image, MSStructureTensorImage &_MSST_image, const string& _winName, Mat& _bgdModel, Mat& _fgdModel , Mat& _MSST_bgdModel, Mat& _MSST_fgdModel );
    void showImage() const;
    int nextIter(int max_iterations);
    int getIterCount() const {
        return iterCount;
    }
    double getXi() const {
        return xi;
    }
    Mat mask;
    Mat initial_mask;
    Mat initial_mask_color;
    Mat initial_mask_msst;
private:
    const string* winName;
    const Mat* image;
    MSStructureTensorImage MSST_image;
    Mat input_mask;
    Mat bgdModel, fgdModel;
    Mat MSST_bgdModel, MSST_fgdModel;

    bool isInitialized;

    int iterCount;

    double xi;
};

void GCApplication::setImageAndWinName( const Mat& _image, MSStructureTensorImage &_MSST_image, const string& _winName, Mat& _bgdModel, Mat& _fgdModel, Mat& _MSST_bgdModel, Mat& _MSST_fgdModel )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    MSST_image = _MSST_image;

    winName = &_winName;

    bgdModel = _bgdModel;
    fgdModel = _fgdModel;
    MSST_bgdModel = _MSST_bgdModel;
    MSST_fgdModel = _MSST_fgdModel;
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

    cg_cmsst_grabCut( *image, MSST_image, mask, initial_mask, initial_mask_color, initial_mask_msst, rect, bgdModel, fgdModel, MSST_bgdModel, MSST_fgdModel, max_iterations, xi );

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
        ("max-iterations,m", po::value<int>()->default_value(10), "maximum number of iterations")
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

    Mat fgdModel, bgdModel;
    Mat input_fgdModel, input_bgdModel;
    FileStorage fs(model_filename, FileStorage::READ);
    fs["fgdModel"] >> fgdModel;
    fs["bgdModel"] >> bgdModel;

    double var_bgd_kl_sym, var_bgd_kl_mr, var_bgd_kl_rm, var_fgd_kl_sym, var_fgd_kl_mr, var_fgd_kl_rm;
    fs["var_bgd_kl_sym"] >> var_bgd_kl_sym;
    fs["var_bgd_kl_mr"] >> var_bgd_kl_mr;
    fs["var_bgd_kl_rm"] >> var_bgd_kl_rm;
    fs["var_fgd_kl_sym"] >> var_fgd_kl_sym;
    fs["var_fgd_kl_mr"] >> var_fgd_kl_mr;
    fs["var_fgd_kl_rm"] >> var_fgd_kl_rm;

    Mat MSST_fgdModel, MSST_bgdModel;
    Mat MSST_input_fgdModel, MSST_input_bgdModel;
    fs["MSST_fgdModel"] >> MSST_fgdModel;
    fs["MSST_bgdModel"] >> MSST_bgdModel;

    fs.release();

    input_fgdModel = fgdModel.clone();
    input_bgdModel = bgdModel.clone();

    MSST_input_fgdModel = MSST_fgdModel.clone();
    MSST_input_bgdModel = MSST_bgdModel.clone();

    Mat image, mask;
    readImageAndMask(input_image, image, mask);
    MSStructureTensorImage MSST_image(image);

    const string winName = "image";
    if(interactive)
        cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );

    gcapp.setImageAndWinName( image, MSST_image, winName, bgdModel, fgdModel, MSST_bgdModel, MSST_fgdModel );
    if(interactive)
        gcapp.showImage();

    if(interactive)
        cvWaitKey(0);

    int iterCount = gcapp.getIterCount();
    cout << "<" << iterCount << "... ";
    int newIterCount = gcapp.nextIter(max_iterations);
    if( newIterCount > iterCount )
    {
        if(interactive)
            gcapp.showImage();
        cout << newIterCount << ">" << endl;
    }

    FileStorage fs2(output_filename, FileStorage::WRITE);
    fs2 << "input_image" << input_image;
    fs2 << "mask" << gcapp.mask;
    fs2 << "initial_mask" << gcapp.initial_mask;
    fs2 << "initial_mask_color" << gcapp.initial_mask_color;
    fs2 << "initial_mask_msst" << gcapp.initial_mask_msst;
    fs2 << "fgdModel" << fgdModel;
    fs2 << "bgdModel" << bgdModel;
    fs2 << "MSST_fgdModel" << MSST_fgdModel;
    fs2 << "MSST_bgdModel" << MSST_bgdModel;

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
        GMM i; i.setModel(input_fgdModel);
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
        GMM i; i.setModel(input_bgdModel);
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

    {
        MSST_GMM i; i.setModel(MSST_input_fgdModel);
        MSST_GMM r; r.setModel(MSST_fgdModel);
        double kl_div_i_r = i.KLdiv(r);
        double kl_div_r_i = r.KLdiv(i);
        double kl_sym = i.KLsym(r);

        cout << " ,msst fgd KL input result: " << kl_div_i_r;
        cout << " ,msst prob fgd KL input result: " << compute_probability( kl_div_i_r, var_fgd_kl_mr);
        cout << " ,msst fgd KL result input: " << kl_div_r_i;
        cout << " ,msst prob fgd KL result input: " << compute_probability( kl_div_r_i, var_fgd_kl_rm);
        cout << " ,msst fgd KL sym: " << kl_sym;
        cout << " ,msst prob fgd KL sym: " << compute_probability( kl_sym, var_fgd_kl_sym);
        fs2 << "msst prob fgd KL sym" << compute_probability( kl_sym, var_fgd_kl_sym);
    }


    {
        MSST_GMM i; i.setModel(MSST_input_bgdModel);
        MSST_GMM r; r.setModel(MSST_bgdModel);
        double kl_div_i_r = i.KLdiv(r);
        double kl_div_r_i = r.KLdiv(i);
        double kl_sym = i.KLsym(r);

        cout << " ,msst bgd KL input result: " << kl_div_i_r;
        cout << " ,msst prob bgd KL input result: " << compute_probability( kl_div_i_r, var_bgd_kl_mr);
        cout << " ,msst bgd KL result input: " << kl_div_r_i;
        cout << " ,msst prob bgd KL result input: " << compute_probability( kl_div_r_i, var_bgd_kl_rm);
        cout << " ,msst bgd KL sym: " << kl_sym;
        cout << " ,msst prob bgd KL sym: " << compute_probability( kl_sym, var_bgd_kl_sym);
    }

    cout << " ,xi: " << gcapp.getXi();

    cout << endl;
    if(interactive)
        cvWaitKey(0);

exit_main:
    if(interactive)
        cvDestroyWindow( winName.c_str() );
    return 0;
}
