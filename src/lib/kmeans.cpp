static void generateRandomCenter(const vector<Vec2f>& box, float* center, RNG& rng)
{
    size_t j, dims = box.size();
    float margin = 1.f/dims;
    for( j = 0; j < dims; j++ )
        center[j] = ((float)rng*(1.f+margin*2.f)-margin)*(box[j][1] - box[j][0]) + box[j][0];
}


double kmeans( const Mat& data, int K, Mat& best_labels,
               TermCriteria criteria, int attempts,
               int flags, Mat* _centers )
{
    const int SPP_TRIALS = 3;
    int N = data.rows > 1 ? data.rows : data.cols;
    int dims = (data.rows > 1 ? data.cols : 1)*data.channels();
    int type = data.depth();
    bool simd = checkHardwareSupport(CV_CPU_SSE);

    attempts = std::max(attempts, 1);
    CV_Assert( data.dims <= 2 && type == CV_32F && K > 0 );

    Mat _labels;
    if( flags & CV_KMEANS_USE_INITIAL_LABELS )
    {
        CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
                   best_labels.cols*best_labels.rows == N &&
                   best_labels.type() == CV_32S &&
                   best_labels.isContinuous());
        best_labels.copyTo(_labels);
    }
    else
    {
        if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
                best_labels.cols*best_labels.rows == N &&
                best_labels.type() == CV_32S &&
                best_labels.isContinuous()))
            best_labels.create(N, 1, CV_32S);
        _labels.create(best_labels.size(), best_labels.type());
    }
    int* labels = _labels.ptr<int>();

    Mat centers(K, dims, type), old_centers(K, dims, type);
    vector<int> counters(K);
    vector<Vec2f> _box(dims);
    Vec2f* box = &_box[0];

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

    const float* sample = data.ptr<float>(0);
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
    }

    for( a = 0; a < attempts; a++ )
    {
        double max_center_shift = DBL_MAX;
        for( iter = 0; iter < criteria.maxCount && max_center_shift > criteria.epsilon; iter++ )
        {
            swap(centers, old_centers);

            if( iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) )
            {
                if( flags & KMEANS_PP_CENTERS )
                    generateCentersPP(data, centers, K, rng, SPP_TRIALS);
                else
                {
                    for( k = 0; k < K; k++ )
                        generateRandomCenter(_box, centers.ptr<float>(k), rng);
                }
            }
            else
            {
                if( iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS) )
                {
                    for( i = 0; i < N; i++ )
                        CV_Assert( (unsigned)labels[i] < (unsigned)K );
                }

                // compute centers
                centers = Scalar(0);
                for( k = 0; k < K; k++ )
                    counters[k] = 0;

                for( i = 0; i < N; i++ )
                {
                    sample = data.ptr<float>(i);
                    k = labels[i];
                    float* center = centers.ptr<float>(k);
                    for( j = 0; j <= dims - 4; j += 4 )
                    {
                        float t0 = center[j] + sample[j];
                        float t1 = center[j+1] + sample[j+1];

                        center[j] = t0;
                        center[j+1] = t1;

                        t0 = center[j+2] + sample[j+2];
                        t1 = center[j+3] + sample[j+3];

                        center[j+2] = t0;
                        center[j+3] = t1;
                    }
                    for( ; j < dims; j++ )
                        center[j] += sample[j];
                    counters[k]++;
                }

                if( iter > 0 )
                    max_center_shift = 0;

                for( k = 0; k < K; k++ )
                {
                    float* center = centers.ptr<float>(k);
                    if( counters[k] != 0 )
                    {
                        float scale = 1.f/counters[k];
                        for( j = 0; j < dims; j++ )
                            center[j] *= scale;
                    }
                    else
                        generateRandomCenter(_box, center, rng);

                    if( iter > 0 )
                    {
                        double dist = 0;
                        const float* old_center = old_centers.ptr<float>(k);
                        for( j = 0; j < dims; j++ )
                        {
                            double t = center[j] - old_center[j];
                            dist += t*t;
                        }
                        max_center_shift = std::max(max_center_shift, dist);
                    }
                }
            }

            // assign labels
            compactness = 0;
            for( i = 0; i < N; i++ )
            {
                sample = data.ptr<float>(i);
                int k_best = 0;
                double min_dist = DBL_MAX;

                for( k = 0; k < K; k++ )
                {
                    const float* center = centers.ptr<float>(k);
                    double dist = distance(sample, center, dims, simd);

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
            if( _centers )
                centers.copyTo(*_centers);
            _labels.copyTo(best_labels);
        }
    }

    return best_compactness;
}

