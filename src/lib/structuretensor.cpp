#include "structuretensor.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

StructureTensorImage::StructureTensorImage(const cv::Mat& image, double sigma)
{
    // Compute Image gradient in both directions for each channel
    cv::Mat xgrad;
    cv::Sobel(image, xgrad, CV_64F, 1, 0, 3);

    cv::Mat ygrad;
    cv::Sobel(image, ygrad, CV_64F, 0, 1, 3);


    // compute structure tensor for each pixel
    cv::Mat stmat(image.rows, image.cols, CV_64FC3, Scalar(0,0,0));

    cv::Point p;
    for( p.y = 0; p.y < stmat.rows; p.y++ )
    {
        for( p.x = 0; p.x < stmat.cols; p.x++ )
        {
            for(int c=0;c<image.channels();c++)
            {
                stmat.at<Vec3d>(p)[0] += xgrad.at<Vec3d>(p)[c]*xgrad.at<Vec3d>(p)[c];
                stmat.at<Vec3d>(p)[1] += ygrad.at<Vec3d>(p)[c]*ygrad.at<Vec3d>(p)[c];
                stmat.at<Vec3d>(p)[2] += xgrad.at<Vec3d>(p)[c]*ygrad.at<Vec3d>(p)[c];
            }
        }
    }

    // average on some of the pixels <- size of the window = scale?
    int ksize = ((int)(4*sigma)) * 2 + 1;
    GaussianBlur(stmat, blurredstmat, Size(ksize, ksize), sigma);

//    blurredstmat = stmat;

    tensors.reserve(stmat.rows * stmat.cols);
    for( p.y = 0; p.y < stmat.rows; p.y++ )
    {
        for( p.x = 0; p.x < stmat.cols; p.x++ )
        {
            tensors.push_back( StructureTensor( blurredstmat.at<Vec3d>(p) ) );
        }
    }
}

cv::Mat StructureTensorImage::getImage() {
    return blurredstmat;
}

StructureTensor::StructureTensor() {
    st = Mat::zeros(2,2,CV_64FC1);
}

StructureTensor::StructureTensor(const cv::Vec3d& t) {
    st.create(2,2,CV_64FC1);
    st.at<double>(0,0) = t[0];
    st.at<double>(1,1) = t[1];
    st.at<double>(0,1) = t[2];
    st.at<double>(1,0) = t[2];
}

StructureTensor::StructureTensor(const cv::Mat& m) {
    assert(m.cols == 2 && m.rows == 2 && m.type() == CV_64FC1);
    st = m.clone();
}

cv::Mat StructureTensor::getMatrix() {
    return st;
}

cv::vector<StructureTensor> StructureTensorImage::getCenters()
{
    return best_centers;
}
cv::Mat StructureTensorImage::getLabels()
{
    int kMeansItCount = 5;
    kmeans(5, TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 2);
    return best_labels;
}

void matrix_sqrt(const Mat& src, Mat& res)
{
    Mat eigenvalues, eigenvectors;
    eigen(src, eigenvalues, eigenvectors);
    
    Mat evec_inv;
    invert(eigenvectors, evec_inv);

    Mat eval_sqrt;
    sqrt(eigenvalues, eval_sqrt);

    Mat eval_mat(2,2, CV_64FC1, Scalar(0.0));
    eval_mat.at<double>(0,0) = eval_sqrt.at<double>(0,0);
    eval_mat.at<double>(1,1) = eval_sqrt.at<double>(1,0);

    res = eigenvectors * eval_mat * evec_inv;
}

double distance2(StructureTensor& stl, StructureTensor& str)
{
    Mat l = stl.getMatrix();
    Mat r = str.getMatrix();
    Mat l_inv; 
    invert( l , l_inv);
    Mat r_inv;
    invert( r, r_inv );

    double res = sqrt( (trace(l_inv * r + r_inv * l)[0] - 4.0)/4.0 );

    return res;

/*    Mat diff = l - r;
    double dist = sqrt( diff.at<double>(0,0) * diff.at<double>(0,0) + diff.at<double>(1,1) * diff.at<double>(1,1) + diff.at<double>(1,0) * diff.at<double>(1,0) );

    return dist;*/
}

StructureTensor mean(std::vector<StructureTensor>& list){

    Mat A(2,2,CV_64FC1,Scalar(0));
    Mat B(2,2,CV_64FC1,Scalar(0));

    for(int i=0; i<list.size(); i++)
    {
        Mat st = list[0].getMatrix();
        A += st;
        Mat st_inv;
        invert(st, st_inv);
        B += st_inv;
    }
    
    A = A / list.size();
    B = B / list.size();

    Mat B_sqrt;
    matrix_sqrt(B, B_sqrt);

    Mat B_inv;
    invert(B, B_inv);

//    cout << "B_inv = " << B_inv.at<double>(0,0) << " " << B_inv.at<double>(1,0) << " " << B_inv.at<double>(0,1) << " " << B_inv.at<double>(1,1) << endl;

    Mat B_inv_sqrt;
    matrix_sqrt(B_inv, B_inv_sqrt);
//    cout << "B_inv_sqrt = " << B_inv_sqrt.at<double>(0,0) << " " << B_inv_sqrt.at<double>(1,0) << " " << B_inv_sqrt.at<double>(0,1) << " " << B_inv_sqrt.at<double>(1,1) << endl;

    Mat BAB = B_sqrt * A * B_sqrt;

    Mat BAB_sqrt;
    matrix_sqrt(BAB, BAB_sqrt);

    Mat res = B_inv_sqrt * BAB_sqrt * B_inv_sqrt;

//    cout << "mean = " << res.at<double>(0,0) << " " << res.at<double>(1,0) << " " << res.at<double>(0,1) << " " << res.at<double>(1,1) << endl;
//    return res;
    Mat sum(2,2,CV_64FC1, Scalar(0.0));

    for(int i=0; i<list.size(); i++)
    {
        //sum = sum + list[i];
        add(sum, list[i].getMatrix(), sum);
    }

    Mat res2 = (sum / list.size());
    
    return res2;

}

static void generateRandomCenter(StructureTensor &center, vector<StructureTensor> tensors, RNG& rng)
{
    center = tensors[rng.uniform(0, tensors.size())];
}

double StructureTensorImage::kmeans( int K, TermCriteria criteria, int attempts)
{
    int N = tensors.size();
//    int dims = (data.rows > 1 ? data.cols : 1)*data.channels();
//    int type = data.depth();
//    bool simd = checkHardwareSupport(CV_CPU_SSE);

    attempts = std::max(attempts, 1);
    CV_Assert( K > 0 );

    Mat _labels;

    if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
            best_labels.cols*best_labels.rows == N &&
            best_labels.type() == CV_32S &&
            best_labels.isContinuous()))
        best_labels.create(N, 1, CV_32S);
    _labels.create(best_labels.size(), best_labels.type());

    int* labels = _labels.ptr<int>();

    vector<StructureTensor> centers(K), old_centers(K);
    vector<int> counters(K);
//    vector<Vec2f> _box(dims);
//    Vec2f* box = &_box[0];

    double best_compactness = DBL_MAX, compactness = 0;
    RNG& rng = theRNG();
    int a, iter, i, j, k;

    if( criteria.type & TermCriteria::EPS )
        criteria.epsilon = std::max(criteria.epsilon, 0.);
    else
        criteria.epsilon = FLT_EPSILON;
    criteria.epsilon *= criteria.epsilon;

    if( criteria.type & TermCriteria::COUNT )
        criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
    else
        criteria.maxCount = 100;

    if( K == 1 )
    {
        attempts = 1;
        criteria.maxCount = 2;
    }

/*    const float* sample = data.ptr<float>(0);
    for( j = 0; j < dims; j++ )
        box[j] = Vec2f(sample[j], sample[j]);

    for( i = 1; i < N; i++ )
    {
        sample = data.ptr<float>(i);
        for( j = 0; j < dims; j++ )
        {
            float v = sample[j];
            box[j][0] = std::min(box[j][0], v);
            box[j][1] = std::max(box[j][1], v);
        }
    }*/

    for( a = 0; a < attempts; a++ )
    {
        double max_center_shift = DBL_MAX;
        for( iter = 0; iter < criteria.maxCount && max_center_shift > criteria.epsilon; iter++ )
        {
            swap(centers, old_centers);

            if( iter == 0 )
            {
                for( k = 0; k < K; k++ )
                    generateRandomCenter(centers[k], tensors, rng);
            }
            else
            {
                // compute centers
                vector< vector<StructureTensor> > clustered_tensors(K);

                // sum up all centers labelwise
                for( i = 0; i < N; i++ )
                {
                    clustered_tensors[labels[i]].push_back( tensors[i] );
                }

                if( iter > 0 )
                    max_center_shift = 0;

                // divide sum by element count
                for( k = 0; k < K; k++ )
                {
                    //cout << "clustered_tensors["<<k<<"].size() = " << clustered_tensors[k].size() << endl;
                    if( clustered_tensors[k].size() != 0 )
                        centers[k] = mean(clustered_tensors[k]);
                    else
                        generateRandomCenter(centers[k], tensors, rng);

                    if( iter > 0 )
                    {
                        StructureTensor l = centers[k];
                        StructureTensor r = old_centers[k];
                        double dist = distance2(l, r);
                        max_center_shift = std::max(max_center_shift, dist);
                    }
                }
                //cout << "max_center_shift = " << max_center_shift << endl;
            }

            // assign labels
            compactness = 0;
            for( i = 0; i < N; i++ )
            {
                int k_best = 0;
                double min_dist = DBL_MAX;

                for( k = 0; k < K; k++ )
                {
                    double dist = distance2(tensors[i], centers[k]);

                    if( min_dist > dist )
                    {
                        min_dist = dist;
                        k_best = k;
                    }
                }

                compactness += min_dist;
                labels[i] = k_best;
            }
        }

        if( compactness < best_compactness )
        {
            best_compactness = compactness;
            best_centers = centers;
            _labels.copyTo(best_labels);
        }
        cout << "compactness = " << compactness << endl;
    }

    return best_compactness;
}

