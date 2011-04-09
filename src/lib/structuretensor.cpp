#include "structuretensor.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <math.h>

#include <time.h>

using namespace cv;
using namespace std;

vector<StructureTensor> StructureTensorImage::getAllTensors() const
{
    return tensors;
}

StructureTensor StructureTensorImage::getTensor(int x, int y) const
{
    assert( y < rows);
    assert( x < cols);
    assert( y >= 0);
    assert( x >= 0);
    return tensors[x * blurredstmat.rows + y];
}

MSStructureTensorImage::MSStructureTensorImage(const cv::Mat& image)
{
    double sigma = 3.0;
    tensors.resize(image.cols * image.rows);
    cols = image.cols;
    rows = image.rows;

    for(;sigma < 10; sigma += 1.0)
    {
        StructureTensorImage sti(image, sigma);
        const vector<StructureTensor> &st = sti.getAllTensors();

        for(int i=0;i<st.size();i++)
        {
            tensors[i].push_back(st[i]);
        }
    }
}

vector<vector<StructureTensor> > MSStructureTensorImage::getAllTensors() const
{
    return tensors;
}

vector<StructureTensor> MSStructureTensorImage::getTensor(int x, int y) const
{
    assert( y < rows);
    assert( x < cols);
    assert( y >= 0);
    assert( x >= 0);
//    return tensors[x * rows + y];
    return tensors[y * cols + x];
}


StructureTensorImage::StructureTensorImage(const cv::Mat& image, double sigma)
{
    // Compute Image gradient in both directions for each channel
    cv::Mat xgrad;
    cv::Sobel(image, xgrad, CV_64F, 1, 0, 3);

    cv::Mat ygrad;
    cv::Sobel(image, ygrad, CV_64F, 0, 1, 3);

    cols = image.cols;
    rows = image.rows;


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

StructureTensor::StructureTensor(double p00, double p11, double p01_10)
{
    st.create(2,2,CV_64FC1);
    st.at<double>(0,0) = p00;
    st.at<double>(1,1) = p11;
    st.at<double>(0,1) = p01_10;
    st.at<double>(1,0) = p01_10;
}


StructureTensor::StructureTensor(const cv::Mat& m) {
    assert(m.cols == 2 && m.rows == 2 && m.type() == CV_64FC1);
    st = m.clone();
}

cv::Mat StructureTensor::getMatrix() const {
    return st;
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

double distance2(const StructureTensor& stl, const StructureTensor& str)
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

double MS_distance2(const vector<StructureTensor>& stl, const vector<StructureTensor>& str)
{
    assert(stl.size() == str.size());

    double sum = 0.0;

    for(int s = 0; s < stl.size(); s++)
    {

        Mat l = stl[s].getMatrix();
        Mat r = str[s].getMatrix();
        Mat l_inv; 
        invert( l , l_inv);
        Mat r_inv;
        invert( r, r_inv );

        sum += (trace(l_inv * r + r_inv * l)[0] - 4.0)/4.0;
    }

    double res = sqrt( sum );

    return res;
}

StructureTensor compute_mean(const std::vector<StructureTensor>& list){

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
/*    Mat sum(2,2,CV_64FC1, Scalar(0.0));

    for(int i=0; i<list.size(); i++)
    {
        //sum = sum + list[i];
        add(sum, list[i].getMatrix(), sum);
    }

    Mat res2 = (sum / list.size());*/
    
    return res;

}

vector<StructureTensor> MS_compute_mean(const std::vector<std::vector<StructureTensor> >& list){
    Mat st_inv;
    Mat st;
    Mat B_sqrt;
    Mat B_inv;
    Mat B_inv_sqrt;
    Mat BAB_sqrt;
    Mat BAB;

    vector<StructureTensor> res;
    for(int s=0;s<list[0].size();s++)
    {
        Mat A(2,2,CV_64FC1,Scalar(0));
        Mat B(2,2,CV_64FC1,Scalar(0));

        for(int i=0; i<list.size(); i++)
        {
            st = list[0][s].getMatrix();
            A += st;
            invert(st, st_inv);
            B += st_inv;
        }
        
        A = A / list.size();
        B = B / list.size();

        matrix_sqrt(B, B_sqrt);

        invert(B, B_inv);

    //    cout << "B_inv = " << B_inv.at<double>(0,0) << " " << B_inv.at<double>(1,0) << " " << B_inv.at<double>(0,1) << " " << B_inv.at<double>(1,1) << endl;

        matrix_sqrt(B_inv, B_inv_sqrt);
    //    cout << "B_inv_sqrt = " << B_inv_sqrt.at<double>(0,0) << " " << B_inv_sqrt.at<double>(1,0) << " " << B_inv_sqrt.at<double>(0,1) << " " << B_inv_sqrt.at<double>(1,1) << endl;

        BAB = B_sqrt * A * B_sqrt;

        matrix_sqrt(BAB, BAB_sqrt);

        res.push_back(StructureTensor(B_inv_sqrt * BAB_sqrt * B_inv_sqrt));
    }
    return res;
}

static void generateRandomCenter(StructureTensor &center, vector<StructureTensor> tensors, RNG& rng)
{
    center = tensors[rng.uniform(0, tensors.size())];
}

static void generateRandomCenter(vector<StructureTensor> &center, vector<vector<StructureTensor> > tensors, RNG& rng)
{
    cout << "generate random center... " << endl;
    center = tensors[rng.uniform(0, tensors.size())];
}


static void generateRandomCenter_kmeanspp(vector<vector<StructureTensor> > &centers, int regenerate_index, vector<vector<StructureTensor> > tensors, RNG& rng)
{
    assert(centers.size() > 0);
    assert(centers.size() > regenerate_index);

    if( regenerate_index == 0 )
        centers[0] = tensors[rng.uniform(0, tensors.size())];
    else {
        vector<double> min_dist(tensors.size());
        for(int t=0; t<tensors.size(); t++)
        {

            min_dist[t] = 10000000000.0; //TODO: double max einsetzen
            for(int j=0; j<regenerate_index; j++)
            {

                double dist = MS_distance2(tensors[t], centers[j]);
                if( dist < min_dist[t] )
                    min_dist[t] = dist;
            }
        }

        vector<double> partial_sum_squares(tensors.size());
        partial_sum_squares[0] = min_dist[0] * min_dist[0];

        for(int j=1; j< tensors.size(); j++)
        {
            partial_sum_squares[j] = partial_sum_squares[j-1] + min_dist[j] * min_dist[j];
        }

        double rand = rng.uniform(0.0, partial_sum_squares.back());

        // find element
        int index = 0;
        while( partial_sum_squares[index] < rand )
            index++;        

        centers[regenerate_index] = tensors[index];
    }
}

static void generateRandomCenters_kmeanspp(vector<vector<StructureTensor> > &centers, vector<vector<StructureTensor> > tensors, RNG& rng)
{
    assert(centers.size() > 0);

    centers[0] = tensors[rng.uniform(0, tensors.size())];

    for(int i=1; i<centers.size(); i++)
    {
        vector<double> min_dist(tensors.size());
        for(int t=0; t<tensors.size(); t++)
        {
            min_dist[t] = 10000000000.0; //TODO: double max einsetzen
            for(int j=0; j<i; j++)
            {
                double dist = MS_distance2(tensors[t], centers[j]);
                if( dist < min_dist[t] )
                    min_dist[t] = dist;
            }
        }

        vector<double> partial_sum_squares(tensors.size());
        partial_sum_squares[0] = min_dist[0] * min_dist[0];

        for(int j=1; j< tensors.size(); j++)
        {
            partial_sum_squares[j] = partial_sum_squares[j-1] + min_dist[j] * min_dist[j];
        }

        double rand = rng.uniform(0.0, partial_sum_squares.back());

        // find element
        int index = 0;
        while( partial_sum_squares[index] < rand )
            index++;        

        centers[i] = tensors[index];
    }
}


double kmeans(const std::vector<StructureTensor> &tensors, int K, cv::TermCriteria criteria, int attempts, cv::Mat &best_labels, std::vector<StructureTensor> &best_centers)
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
                        centers[k] = compute_mean(clustered_tensors[k]);
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

double MSST_kmeans(const std::vector<std::vector<StructureTensor> > &tensors, int K, cv::TermCriteria criteria, int attempts, cv::Mat &best_labels, std::vector<std::vector<StructureTensor> > &best_centers)
{
    clock_t start, finish;
    start = clock();


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

    vector<vector<StructureTensor> > centers(K), old_centers(K);
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
//            old_centers = centers;

            if( iter == 0 )
            {
                generateRandomCenters_kmeanspp(centers, tensors, rng);
                //for( k = 0; k < K; k++ )
                //    generateRandomCenter(centers[k], tensors, rng);
            }
            else
            {
                // compute centers
                vector< vector< vector<StructureTensor> > > clustered_tensors(K);

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
                        centers[k] = MS_compute_mean(clustered_tensors[k]);
                    else
                        //generateRandomCenter(centers[k], tensors, rng);
                        generateRandomCenter_kmeanspp(centers, k, tensors, rng);

                    if( iter > 0 )
                    {
                        vector<StructureTensor> l = centers[k];
                        vector<StructureTensor> r = old_centers[k];
                        double dist = MS_distance2(l, r);
                        max_center_shift = std::max(max_center_shift, dist);
                    }
                }
                cout << "max_center_shift = " << max_center_shift << endl;
            }

            // assign labels
            compactness = 0;
            for( i = 0; i < N; i++ )
            {
                int k_best = 0;
                double min_dist = DBL_MAX;

                for( k = 0; k < K; k++ )
                {
                    double dist = MS_distance2(tensors[i], centers[k]);

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

    finish = clock();

    cout << "time: " << ( (double)(finish - start)/CLOCKS_PER_SEC ) << std::endl;

    return best_compactness;
}



double compute_variance(const std::vector<StructureTensor>& list)
{
    StructureTensor mean = compute_mean(list);

    double sum_dist = 0.0;
    for(int i=0;i<list.size();i++)
    {
        double dist = distance2(list[i], mean);
        sum_dist += dist * dist;
    }
    return sum_dist / list.size();
}

double MS_compute_variance(const std::vector<std::vector<StructureTensor> >& list)
{
    vector<StructureTensor> mean = MS_compute_mean(list);

    double sum_dist = 0.0;
    for(int i=0;i<list.size();i++)
    {
        double dist = MS_distance2(list[i], mean);
        sum_dist += dist * dist;
    }
    return sum_dist / list.size();
}
