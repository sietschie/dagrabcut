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
        ("model file", po::value<string>()->required(), "the model file to test with")
        ("algorithm results", po::value<vector<string> >()->required(), "the files containing the output masks")
    ;

    po::positional_options_description positional;
    positional.add("model file", 1);
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
        cout << "Usage: ./shapes.bin [options] outputfile resultfile1 resultfile2 ... \n";
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

    string model_filename = vm["model file"].as< string >();
    vector<string> filenames = vm["algorithm results"].as< vector<string> >();

    // read in masks from model file
    vector<Vec<double,7> > huMoments_train;
    vector<int> responses_train;

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

        huMoments_train.push_back(huMoment);
        responses_train.push_back(class_number);
    }
    fs_model.release();

    // compute mean, variance
    Vec<double, 7> mean = computeMean( huMoments_train );
    Vec<double, 7> variances = computeVariance(huMoments_train);

    // compute knn-distances
    // vector to matrix
    Mat huMomentsMatrix(huMoments_train.size(), 7, CV_32FC1);
    Mat responsesMatrix(responses_train.size(), 1, CV_32SC1);

    for(int x=0;x<huMoments_train.size();x++)
    {
        for(int y=0;y<7;y++)
        {
            double tmp = huMoments_train[x][y];
            huMomentsMatrix.at<float>(x,y) = tmp;
        }
        responsesMatrix.at<int>(x) = responses_train[x];
    }


    CvKNearest knn;
    knn.train(huMomentsMatrix, responsesMatrix);

    for(vector<string>::iterator itr = filenames.begin(); itr != filenames.end(); itr++)
    {
        FileStorage fs(*itr, FileStorage::READ);

        Mat mask;
        fs["mask"] >> mask;
        fs.release();

        Vec<double, 7> huMoment;
        computeHuMoments(mask, huMoment);

        Mat inputMatrix(1, 7, CV_32FC1);
        for(int y=0;y<7;y++)
            inputMatrix.at<float>(0,y) = huMoment[y];
        //fs2 << "huMoments" << huMomentsMatrix;

        FileStorage fs2(*itr, FileStorage::APPEND);

        Mat dist;
        knn.find_nearest(inputMatrix, 1, NULL, NULL, NULL, &dist);

        cout << "dist humoments 1nn: " << dist.at<float>(0,0);
        fs2 << "dist humoments 1nn" << dist.at<float>(0,0);

        knn.find_nearest(inputMatrix, 3, NULL, NULL, NULL, &dist);
        cout << ", dist humoments 3nn: " << dist.at<float>(0,0) + dist.at<float>(0,1) + dist.at<float>(0,2);
        fs2 << "dist humoments 3nn" << dist.at<float>(0,0) + dist.at<float>(0,1) + dist.at<float>(0,2);


        double prob_humoments = computeProbability( huMoment, mean, variances);

        fs2 << "prob humoments" << prob_humoments;
        fs2.release();

        cout << ", prob humoments: " << prob_humoments;

        cout << endl;
    }

    return 0;
}
