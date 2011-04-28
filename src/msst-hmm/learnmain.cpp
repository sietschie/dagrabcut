#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <boost/program_options.hpp>

#include "../lib/msst_hmm.hpp"
#include "../lib/hmm.hpp"
#include "../lib/structuretensor.hpp"
#include "../lib/shared.hpp"

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
    MSST_HMM MSST_fgdHmm, MSST_bgdHmm;
    vector<HMM> fgdHmms, bgdHmms;
    vector<MSST_HMM> MSST_fgdHmms, MSST_bgdHmms;

    for(vector<string>::iterator filename = input_images.begin(); filename != input_images.end(); ++filename)    
    {
        vector<Vec3f> bgdSamples, fgdSamples;
        vector<vector<StructureTensor> > MSST_bgdSamples, MSST_fgdSamples;
        HMM cur_fgdHmm, cur_bgdHmm;
        MSST_HMM MSST_cur_fgdHmm, MSST_cur_bgdHmm;

        Mat image, mask;
        readImageAndMask(*filename, image, mask);

        MSStructureTensorImage stimage(image);

        Mat bgdModel, fgdModel;
        computeGMM(*filename, image, mask, model_filename, class_number, bgdModel, fgdModel);

        Mat MSST_bgdModel, MSST_fgdModel;
        MSST_computeGMM(*filename, stimage, mask, model_filename, class_number, MSST_bgdModel, MSST_fgdModel);

        Mat binary_mask = mask.clone();

        Point p;
        for( p.y = 0; p.y < image.rows; p.y++ )
        {
            for( p.x = 0; p.x < image.cols; p.x++ )
            {
                binary_mask.at<uchar>(p) = mask.at<uchar>(p) == class_number ? 1 : 0;
            }
        }

        //Mat binary_mask = mask & 1;
        //Mat binary_mask = mask & class_number;
        cur_fgdHmm.setModel(fgdModel, binary_mask, image);
        cur_bgdHmm.setModel(bgdModel, 1 - binary_mask, image);

        MSST_cur_fgdHmm.setModel(MSST_fgdModel, binary_mask, stimage);
        MSST_cur_bgdHmm.setModel(MSST_bgdModel, 1 - binary_mask, stimage);

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

        MSST_fgdHmms.push_back(MSST_cur_fgdHmm);
        MSST_bgdHmms.push_back(MSST_cur_bgdHmm);
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
        fgdHmm.addModel(fgdHmms[i]);
    for(int i=0; i<bgdHmms.size(); i++)
        bgdHmm.addModel(bgdHmms[i]);
    for(int i=0; i<MSST_fgdHmms.size(); i++)
        MSST_fgdHmm.addModel(MSST_fgdHmms[i]);
    for(int i=0; i<MSST_bgdHmms.size(); i++)
        MSST_bgdHmm.addModel(MSST_bgdHmms[i]);

    cout << "KLdiv: " << fgdHmm.KLdiv(bgdHmm) << endl;
    cout << "KLdiv: " << bgdHmm.KLdiv(fgdHmm) << endl;

    fgdHmm.normalize_weights();
    bgdHmm.normalize_weights();
    MSST_fgdHmm.normalize_weights();
    MSST_bgdHmm.normalize_weights();

    while( fgdHmm.getComponentsCount() > vm["cluster"].as<int>())
    {
        fgdHmm.cluster_once();
        bgdHmm.cluster_once();
        MSST_fgdHmm.cluster_once();
        MSST_bgdHmm.cluster_once();
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

    double var_bgd_kl_sym, var_bgd_kl_mr, var_bgd_kl_rm, var_fgd_kl_sym, var_fgd_kl_mr, var_fgd_kl_rm;
    compute_variance(input_images, bgdHmm.getModel(), fgdHmm.getModel(), nr_gaussians, class_number, model_filename,
        var_bgd_kl_sym, var_bgd_kl_mr, var_bgd_kl_rm, var_fgd_kl_sym, var_fgd_kl_mr, var_fgd_kl_rm );

    fs2 << "var_bgd_kl_sym" << var_bgd_kl_sym;
    fs2 << "var_bgd_kl_mr" << var_bgd_kl_mr;
    fs2 << "var_bgd_kl_rm" << var_bgd_kl_rm;
    fs2 << "var_fgd_kl_sym" << var_fgd_kl_sym;
    fs2 << "var_fgd_kl_mr" << var_fgd_kl_mr;
    fs2 << "var_fgd_kl_rm" << var_fgd_kl_rm;

    fs2 << "MSST_fgdHmm" << MSST_fgdHmm;
    fs2 << "MSST_bgdHmm" << MSST_bgdHmm;

    double msst_var_bgd_kl_sym, msst_var_bgd_kl_mr, msst_var_bgd_kl_rm, msst_var_fgd_kl_sym, msst_var_fgd_kl_mr, msst_var_fgd_kl_rm;
    MSST_compute_variance(input_images, MSST_bgdHmm.getModel(), MSST_fgdHmm.getModel(), nr_gaussians, class_number, model_filename,
        msst_var_bgd_kl_sym, msst_var_bgd_kl_mr, msst_var_bgd_kl_rm, msst_var_fgd_kl_sym, msst_var_fgd_kl_mr, msst_var_fgd_kl_rm );

    fs2 << "msst_var_bgd_kl_sym" << msst_var_bgd_kl_sym;
    fs2 << "msst_var_bgd_kl_mr" << msst_var_bgd_kl_mr;
    fs2 << "msst_var_bgd_kl_rm" << msst_var_bgd_kl_rm;
    fs2 << "msst_var_fgd_kl_sym" << msst_var_fgd_kl_sym;
    fs2 << "msst_var_fgd_kl_mr" << msst_var_fgd_kl_mr;
    fs2 << "msst_var_fgd_kl_rm" << msst_var_fgd_kl_rm;

    fs2 << "class_number" << class_number;

    fs2.release();

    return 0;
}
