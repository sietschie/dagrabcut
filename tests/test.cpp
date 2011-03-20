#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MyTest
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <opencv2/core/core.hpp>
#include "../src/gmm.hpp"
#include "../src/hmm.hpp"
#include "../src/gaussian.hpp"

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

BOOST_AUTO_TEST_CASE(MyTestCaseGMM_ComponentsCount)
{
    GMM gmm;
    gmm.setComponentsCount(1);
    BOOST_CHECK_EQUAL( gmm.getComponentsCount(), 1);

    gmm.setComponentsCount(50);
    BOOST_CHECK_EQUAL( gmm.getComponentsCount(), 50);

    gmm.setComponentsCount(5);
    BOOST_CHECK_EQUAL( gmm.getComponentsCount(), 5);
}

BOOST_AUTO_TEST_CASE(MyTestCaseGMM_ModelsInputOutput)
{
    GMM gmm;
    gmm.setModel(randomModel);
    cv::Mat outputModel = gmm.getModel();
    BOOST_CHECK_CLOSE( cv::mean(outputModel)[0], cv::mean(randomModel)[0], 0.0001f);
    BOOST_CHECK_CLOSE(cv::sum(outputModel)[0], cv::sum(randomModel)[0], 0.0001f);
    BOOST_CHECK_EQUAL( gmm.getComponentsCount(), randomModel.cols);

    gmm.setModel(randomModel);
    outputModel = gmm.getModel();
    cv::Mat outputModel_clone = outputModel.clone();
    
    gmm.setModel(randomModel2);
    cv::Mat outputModel2 = gmm.getModel();
    BOOST_CHECK_CLOSE( cv::mean(outputModel)[0], cv::mean(outputModel_clone)[0], 0.0001f);
}


BOOST_AUTO_TEST_CASE(MyTestCaseGMM_ModelsInputOutput2)
{
    // Bug: Input Model was set to Zero after initializing the GMM
    GMM gmm;
    cv::Mat tmp = randomModel.clone();
    gmm.setModel(randomModel);
    BOOST_CHECK_CLOSE( cv::mean(tmp)[0], cv::mean(randomModel)[0], 0.0001f);

}

BOOST_AUTO_TEST_CASE(MyTestCaseGMM_KLDiv)
{
    GMM gmm1, gmm2;
    gmm1.setModel(model_1component_1);
    gmm2.setModel(model_1component_2);

    double kldiv_12 = gmm1.KLdiv(gmm2);
    double kldiv_21 = gmm2.KLdiv(gmm1);

    double klsym_12 = gmm1.KLsym(gmm2);
    double klsym_21 = gmm2.KLsym(gmm1);

    BOOST_CHECK_CLOSE( klsym_12, klsym_21, 0.0001f);


    gmm1.setModel(model_2component_1);
    gmm2.setModel(model_2component_2);
    klsym_12 = gmm1.KLsym(gmm2);
    klsym_21 = gmm2.KLsym(gmm1);
    BOOST_CHECK_CLOSE( klsym_12, klsym_21, 0.0001f);
    BOOST_CHECK_CLOSE( klsym_12, 0.0, 0.0001f);
}

BOOST_AUTO_TEST_CASE(MyTestCaseGaussian_CopyOperator)
{
    Gaussian g1;
    g1.cov = cv::Mat::ones(3,3, CV_64FC1);
    g1.mean = cv::Mat::ones(3,1, CV_64FC1);

    Gaussian g2;
    g2 = g1;
    g2.cov.at<double>(1,1) = 2.0;
    g2.mean.at<double>(1,0) = 3.0;

    BOOST_CHECK_CLOSE( cv::sum(g1.cov)[0], cv::sum(g2.cov)[0] - 1.0, 0.0001f);
    BOOST_CHECK_CLOSE( cv::sum(g1.mean)[0], cv::sum(g2.mean)[0] - 2.0, 0.0001f);
}

BOOST_AUTO_TEST_CASE(MyTestCaseGaussian_CopyConstructor)
{
    Gaussian g1;
    g1.cov = cv::Mat::ones(3,3, CV_64FC1);
    g1.mean = cv::Mat::ones(3,1, CV_64FC1);

    Gaussian g2 = g1;
    g2.cov.at<double>(1,1) = 2.0;
    g2.mean.at<double>(1,0) = 3.0;

    BOOST_CHECK_CLOSE( cv::sum(g1.cov)[0], cv::sum(g2.cov)[0] - 1.0, 0.0001f);
    BOOST_CHECK_CLOSE( cv::sum(g1.mean)[0], cv::sum(g2.mean)[0] - 2.0, 0.0001f);
}

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

BOOST_AUTO_TEST_CASE(MyTestCase)
{
    float x = 9.5f;
    
    BOOST_CHECK(x != 0.0f);
    BOOST_CHECK_EQUAL((int)x, 9);
    BOOST_CHECK_CLOSE(x, 9.5f, 0.0001f);
}

BOOST_AUTO_TEST_SUITE_END()
