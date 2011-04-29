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
        ("algorithm result", po::value<string>()->required(), "the file containing the output masks")
    ;

    po::positional_options_description positional;
    positional.add("output", 1);
    positional.add("algorithm result", 1);

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
        cout << "Usage: ./shapes.bin [options] outputfile resultfile1\n";
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

Vec<double,7> computeMean(vector< Vec<double,7> > &list_of_vectors)
{
    assert( list_of_vectors.size() > 0 );

    vector< Vec<double,7> >::iterator itr = list_of_vectors.begin();
    Vec<double,7> mean;

    for(int i=0;i<7;i++)
        mean[i] = 0.0;

    for(; itr != list_of_vectors.end(); itr++)
    {
        for(int i=0;i<7;i++)
        {
            mean[i] += (*itr)[i];
        }
    }

    for(int i=0;i<7;i++)
    {
        mean[i] /= list_of_vectors.size();
    }

    return mean;
}

Vec<double,7> computeVariance(vector< Vec<double,7> > &list_of_vectors)
{
    assert( list_of_vectors.size() > 0 );

    Vec<double,7> mean = computeMean(list_of_vectors);

    Vec<double,7> sum_of_square_distance;

    for(int i=0;i<7;i++)
        sum_of_square_distance[i] = 0.0;

    vector< Vec<double,7> >::iterator itr = list_of_vectors.begin();
    for(; itr != list_of_vectors.end(); itr++)
    {
        for(int i=0;i<7;i++)
        {
            double dist = mean[i] - (*itr)[i];
            sum_of_square_distance[i] += dist * dist;
        }
    }

    Vec<double,7> variance;
    for(int i=0;i<7;i++)
    {
        variance[i] = sum_of_square_distance[i] / list_of_vectors.size();
    }

    return variance;
}

double computeProbability(Vec<double, 7> huMoments, Vec<double, 7> mean, Vec<double, 7> variance)
{
    const double dim = 7;
    double det = 1.0;

    for(int y=0;y<dim;y++)
    {
        det *= variance[y];
    }

    double factor = 1.0 / pow( 2.0 * M_PI, (double) dim / 2.0) * pow(det, 0.5);

    double sum = 0.0;
    for(int i=0;i< dim;i++)
    {
        double dist = mean[i] - huMoments[i];
        sum += ( dist * dist ) / variance[i];
    }

    double res = factor * pow( factor * exp( sum / (-2.0) ), 1.0 / dim );

    return res;
}


int main( int argc, char** argv )
{
    po::variables_map vm = parseCommandline(argc, argv);

    string filename = vm["algorithm result"].as< string >();
    string output_filename = vm["output"].as<string>();

    vector<Vec<double,7> > huMoments;
    vector<int> responses;

    cout << "open model " << filename << "\n";
    FileStorage fs(filename, FileStorage::READ);

    string model_filename;
    fs["model_filename"] >> model_filename;

    Mat mask;
    fs["mask"] >> mask;

    double class_number;
    fs["class_number"] >> class_number;
    fs.release();

    Vec<double, 7> huMoment;
    computeHuMoments(mask, class_number, huMoment);

    FileStorage fs_model(model_filename, FileStorage::READ);


    FileNodeIterator fni = fs_model["files"].begin();
    for(; fni != fs_model["files"].end(); fni++)
    {
        int class_number;
        fs_model["class_number"] >> class_number;

        string image_filename = *fni;
        cout << "compute hu moments for " << image_filename << "\n";

        Mat image, mask;
        readImageAndMask(image_filename, image, mask);

        Vec<double, 7> huMoment;
        computeHuMoments(mask, class_number, huMoment);

        huMoments.push_back(huMoment);
        responses.push_back(class_number);
    }
    fs_model.release();

/*        Mat image, mask;
        readImageAndMask(filename, image, mask);

        Vec<double,7> huMoment;
        computeHuMoments(mask, class_number, huMoment);
        huMoments.push_back( huMoment );*/

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

    Vec<double, 7> mean = computeMean( huMoments );
    for(int y=0;y<7;y++)
    {
        cout << mean[y] << "  ";
    }
    cout << endl;


    Vec<double, 7> variances = computeVariance(huMoments);
    
    for(vector< Vec<double,7> >::iterator itr = huMoments.begin(); itr != huMoments.end(); itr++)
    {
        cout << "res = " << computeProbability( *itr, mean, variances) << endl;
    }

    FileStorage fs2(filename, FileStorage::APPEND);
    fs2 << "testvariable" << 6;
    fs2.release();

    cout << "-------------------\n";

    cout << " res = " << computeProbability( huMoment, mean, variances) << endl;

    //fs2 << "huMoments" << huMomentsMatrix;

    //CvSVM svm;

    //CvSVMParams params;
    //svm.train_auto(huMomentsMatrix, responsesMatrix, Mat(), Mat(), params, min<int>(10, huMoments.size()));

    //std::cout << "jallo world...\n";
}
