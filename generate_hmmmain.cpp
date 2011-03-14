//#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "hmm.hpp"
#include "grabcut.hpp"

using namespace cv;
using namespace std;

void help()
{
    cout << "Call:\n"
    		"./grabcut <input_image_name1> <input_image_name2>... <class_number> <model_name>\n"
			"reads input_images and input_image_names.yml, generates\n"
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

void learnGMMfromSamples(vector<Vec3f> samples, Mat& model)
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;
    const int componentsCount = 5;

    Mat labels;
    CV_Assert( !samples.empty() );
    Mat _samples( (int)samples.size(), 3, CV_32FC1, &samples[0][0] );

    kmeans( _samples, componentsCount, labels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType, 0 );

    cout << "start learning GMM..." << endl;
    
    GMM gmm(model);

    gmm.initLearning();
    for( int i = 0; i < (int)samples.size(); i++ )
        gmm.addSample( labels.at<int>(i,0), samples[i] );
    gmm.endLearning();
}


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

int main( int argc, char** argv )
{
	if( argc < 3)
	{
		help();
		return 1;
	}

	int class_number = atoi(argv[argc-2]);

    HMM fgdHmm, bgdHmm;
    vector<HMM> fgdHmms, bgdHmms;


    for(int i=1; i<argc-2; i++)
    {
        vector<Vec3f> bgdSamples, fgdSamples;
        HMM cur_fgdHmm, cur_bgdHmm;
        string filename = argv[i];
        
        std::cout << "Reading " << filename << "..." << std::endl;

        Mat image, mask;
        readImageAndMask(filename, image, mask);

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
    for(int i=0;i<fgdHmms.size();i++)
    for(int j=i+1;i<fgdHmms.size();i++)
    {
        fgdDivs.push_back(fgdHmms[i].KLsym(fgdHmms[j]));
    }
	cout << "fgdDivs:  ";
	print_mean_variance(fgdDivs);

    vector<double> bgdDivs;
    for(int i=0;i<bgdHmms.size();i++)
    for(int j=i+1;i<bgdHmms.size();i++)
    {
        bgdDivs.push_back(bgdHmms[i].KLsym(bgdHmms[j]));
    }
	cout << "bgdDivs:  ";
	print_mean_variance(bgdDivs);

    vector<double> betweenDivs;
    for(int i=0;i<bgdHmms.size();i++)
    for(int j=0;i<fgdHmms.size();i++)
    {
        betweenDivs.push_back(bgdHmms[i].KLsym(bgdHmms[j]));
    }
	cout << "betweenDivs:  ";
	print_mean_variance(betweenDivs);


    for(int i=0;i<fgdHmms.size();i++)
        fgdHmm.add_model(fgdHmms[i]);
    for(int i=0;i<bgdHmms.size();i++)
        bgdHmm.add_model(bgdHmms[i]);

    cout << "KLdiv: " << fgdHmm.KLdiv(bgdHmm) << endl;
    cout << "KLdiv: " << bgdHmm.KLdiv(fgdHmm) << endl;

    fgdHmm.normalize_weights();
    bgdHmm.normalize_weights();

    while( fgdHmm.components.size() > 5)
    {
        fgdHmm.cluster_once();
        bgdHmm.cluster_once();
    }

    FileStorage fs2(argv[argc-1], FileStorage::WRITE);
    fs2 << "first" << 2;
    fs2 << "files" << "[";
    for(int i=1; i<argc-1;i++)
    {
        fs2 << argv[i];
    }
    fs2 << "]";
    fs2 << "fgdHmm" << fgdHmm;
    fs2 << "bgdHmm" << bgdHmm;
    fs2.release();

    fgdHmm.free_components();
    bgdHmm.free_components();

/*    HMM testHmm;
    FileStorage fs3("test.yml", FileStorage::READ);
    readHMM(fs3["fgdHmm"], testHmm);
    fs3.release();

    FileStorage fs4("test2.yml", FileStorage::WRITE);
    fs4 << "fgdHmm" << testHmm;
    fs4.release();*/
}
