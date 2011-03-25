#define BOOST_TEST_MODULE MyTestHmm
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <opencv2/core/core.hpp>
#include "../lib/gmm.hpp"
#include "../lib/hmm.hpp"
#include "../lib/gaussian.hpp"

struct Data
{
    cv::Mat Model;
    cv::Mat randomModel;
    cv::Mat randomModel2;
    cv::Mat ones;
    cv::Mat model_1component_1;
    cv::Mat model_1component_2;
    cv::Mat model_2component_1;
    cv::Mat model_2component_2;
    Data()
    {
        BOOST_TEST_MESSAGE("set up data");
        randomModel.create(13,5, CV_64FC1);
        cv::randu(randomModel, Scalar(0), Scalar(256));

        randomModel2.create(13,5, CV_64FC1);
        cv::randu(randomModel2, Scalar(0), Scalar(256));

        ones = cv::Mat::ones(13,5, CV_64FC1);

        model_2component_1 = (Mat_<double>(13,1) << 
                 0.0, 0.0, 0.0,

                 1.0, 0.0, 0.0, 
                 0.0, 1.0, 0.0, 
                 0.0, 0.0, 1.0,

                 1.0,

                 0.5, 0.5, 0.5,

                 2.0, 0.0, 0.0, 
                 0.0, 0.5, 0.0, 
                 0.0, 0.0, 1.0,

                 1.0);

        model_2component_2 = (Mat_<double>(13,1) << 
                 0.5, 0.5, 0.5,

                 2.0, 0.0, 0.0, 
                 0.0, 0.5, 0.0, 
                 0.0, 0.0, 1.0,

                 1.0,

                 0.0, 0.0, 0.0,

                 1.0, 0.0, 0.0, 
                 0.0, 1.0, 0.0, 
                 0.0, 0.0, 1.0,

                 1.0);


        model_1component_1 = (Mat_<double>(13,1) << 
                 0.0, 0.0, 0.0,

                 1.0, 0.0, 0.0, 
                 0.0, 1.0, 0.0, 
                 0.0, 0.0, 1.0,

                 1.0);

        model_1component_2 = (Mat_<double>(13,1) << 
                 0.5, 0.5, 0.5,

                 2.0, 0.0, 0.0, 
                 0.0, 0.5, 0.0, 
                 0.0, 0.0, 1.0,

                 1.0);
    }

    ~Data()
    {
        BOOST_TEST_MESSAGE("tear down data");
    }
};

BOOST_FIXTURE_TEST_SUITE(TestCaseMM, Data)

BOOST_AUTO_TEST_CASE(MyTestCaseHMMComponent_CopyConstructor)
{
    HMM_Component c1;
    c1.left_child = new HMM_Component();
    c1.right_child = new HMM_Component();
    c1.div = 1.0;
    c1.gauss.cov = cv::Mat::ones(3,3, CV_64FC1);
    c1.gauss.mean = cv::Mat::ones(3,1, CV_64FC1);
    c1.weight = 1.234;

    Vec3b p;
    p[0] = 1.0;
    p[1] = 1.1;
    p[2] = 1.2;
    c1.samples.push_back(p);
    HMM_Component c2 = c1;

    BOOST_CHECK(c1.left_child != c2.left_child);
    BOOST_CHECK(c1.right_child != c2.right_child);
    BOOST_CHECK_EQUAL(c1.samples.size(), c2.samples.size());
    BOOST_CHECK_CLOSE(c2.div, c1.div, 0.0001f);

    BOOST_CHECK_CLOSE( cv::sum(c1.gauss.cov)[0], cv::sum(c2.gauss.cov)[0], 0.0001f);
    BOOST_CHECK_CLOSE( cv::sum(c1.gauss.mean)[0], cv::sum(c2.gauss.mean)[0], 0.0001f);
    BOOST_CHECK_CLOSE( c1.weight, c2.weight, 0.0001f);

    BOOST_CHECK_CLOSE( cv::sum(c1.gauss.cov)[0], 9.0 , 0.0001f);
    BOOST_CHECK_CLOSE( cv::sum(c1.gauss.mean)[0], 3.0 , 0.0001f);
    BOOST_CHECK_CLOSE( c1.weight, 1.234, 0.0001f);
}

BOOST_AUTO_TEST_CASE(MyTestCaseHMMComponent_AssignmentOperator)
{
    HMM_Component c1;
    c1.left_child = new HMM_Component();
    c1.right_child = new HMM_Component();
    c1.div = 1.0;
    c1.gauss.cov = cv::Mat::ones(3,3, CV_64FC1);
    c1.gauss.mean = cv::Mat::ones(3,1, CV_64FC1);
    c1.weight = 1.234;
    Vec3b p;
    p[0] = 1.0;
    p[1] = 1.1;
    p[2] = 1.2;
    c1.samples.push_back(p);
    HMM_Component c2;
    c2 = c1;

    BOOST_CHECK(c1.left_child != c2.left_child);
    BOOST_CHECK(c1.right_child != c2.right_child);
    BOOST_CHECK_EQUAL(c1.samples.size(), c2.samples.size());
    BOOST_CHECK_EQUAL(c1.samples.size(), 1);
    BOOST_CHECK_CLOSE(c2.div, c1.div, 0.0001f);

    BOOST_CHECK_CLOSE( cv::sum(c1.gauss.cov)[0], cv::sum(c2.gauss.cov)[0], 0.0001f);
    BOOST_CHECK_CLOSE( cv::sum(c1.gauss.mean)[0], cv::sum(c2.gauss.mean)[0], 0.0001f);
    BOOST_CHECK_CLOSE( c1.weight, c2.weight, 0.0001f);

    BOOST_CHECK_CLOSE( cv::sum(c1.gauss.cov)[0], 9.0 , 0.0001f);
    BOOST_CHECK_CLOSE( cv::sum(c1.gauss.mean)[0], 3.0 , 0.0001f);
    BOOST_CHECK_CLOSE( c1.weight, 1.234, 0.0001f);
}

BOOST_AUTO_TEST_CASE(MyTestCase_FileWrite)
{
    HMM org, res;
    org.setModel(randomModel);
    //org.components[0].div = 1.2;
    //org.components[0].samples.resize(10);

    {
        FileStorage fs("tmp.yml", FileStorage::WRITE);
        fs << "fgdHmm" << org;
        fs.release();
    }

    {    
        FileStorage fs("tmp.yml", FileStorage::READ);
        readHMM(fs["fgdHmm"], res);
        fs.release();
    }

    cv::Mat resModel = res.getModel();
    BOOST_CHECK_CLOSE( cv::mean(resModel)[0], cv::mean(randomModel)[0], 0.0001f);
    BOOST_CHECK_CLOSE(cv::sum(resModel)[0], cv::sum(randomModel)[0], 0.0001f);
    BOOST_CHECK_EQUAL( res.getComponentsCount(), randomModel.cols);
    //BOOST_CHECK_EQUAL( res.components[0].samples.size(), 10);
    //BOOST_CHECK_CLOSE( res.components[0].div, 1.2, 0.0001f);

}


BOOST_AUTO_TEST_SUITE_END()
