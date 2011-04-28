#include <iostream>
#include <boost/program_options.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <algorithm>

#include "../lib/shared.hpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

po::variables_map parseCommandline(int argc, char** argv)
{
    po::options_description generic("Generic options");
    generic.add_options()
        ("help,h", "produce help message")
    ;

    po::options_description hidden("Hidden options");
    hidden.add_options()
        ("output", po::value<string>()->required(), "the output file")
        ("algorithm results", po::value<vector<string> >()->required(), "the files containing the output masks")
    ;

    po::positional_options_description positional;
    positional.add("output", 1);
    positional.add("algorithm results", -1);

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
        cout << "Usage: ./shapes.bin [options] outputfile resultfile1 resultfile2 ...\n";
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

    vector< string > inputs_filenames = vm["algorithm results"].as< vector<string> >();
    string output_filename = vm["output"].as<string>();

    vector<Vec<double,7> > huMoments;
    vector<int> responses;

    for(vector<string>::iterator filename = inputs_filenames.begin(); filename != inputs_filenames.end(); ++filename)    
    {
        cout << "open model " << *filename << "\n";
        FileStorage fs(*filename, FileStorage::READ);

        string model_filename;
        fs["model_filename"] >> model_filename;

        FileStorage fs_model(model_filename, FileStorage::READ);

        int class_number;
        fs_model["class_number"] >> class_number;

        FileNodeIterator fni = fs_model["files"].begin();
        for(; fni != fs_model["files"].end(); fni++)
        {
            string image_filename = *fni;
            cout << "compute hu moments for " << image_filename << "\n";

            Mat image, mask;
            readImageAndMask(image_filename, image, mask);

            Vec<double, 7> huMoment;
            computeHuMoments(mask, class_number, huMoment);

            huMoments.push_back(huMoment);
            responses.push_back(class_number);
        }



/*        Mat image, mask;
        readImageAndMask(filename, image, mask);

        Vec<double,7> huMoment;
        computeHuMoments(mask, class_number, huMoment);
        huMoments.push_back( huMoment );*/

    }

    // vector to matrix
    Mat huMomentsMatrix(huMoments.size(), 7, CV_32FC1);
    Mat responsesMatrix(responses.size(), 1, CV_32SC1);

    for(int x=0;x<huMoments.size();x++)
    {
        for(int y=0;y<7;y++)
        {
            double tmp = huMoments[x][y];
            huMomentsMatrix.at<float>(x,y) = tmp;
            cout << tmp << "  ";
        }
        responsesMatrix.at<int>(x) = responses[x];
        cout << responses[x] << "\n";
    }
    //fs2 << "huMoments" << huMomentsMatrix;

    CvSVM svm;

    CvSVMParams params;
    svm.train_auto(huMomentsMatrix, responsesMatrix, Mat(), Mat(), params, min<int>(10, huMoments.size()));

    std::cout << "jallo world...\n";
}
