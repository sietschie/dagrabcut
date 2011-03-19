#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MyTest
#include <boost/test/unit_test.hpp>
#include <opencv2/core/core.hpp>
#include "../gmm.hpp"

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
        randomModel.create(13,10, CV_64FC1);
        cv::randu(randomModel, Scalar(0), Scalar(256));

        randomModel2.create(13,10, CV_64FC1);
        cv::randu(randomModel2, Scalar(0), Scalar(256));

        ones = cv::Mat::ones(13,10, CV_64FC1);

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
    BOOST_CHECK_EQUAL( gmm.getComponentsCount(), 10);

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

BOOST_AUTO_TEST_CASE(MyTestCase)
{
    float x = 9.5f;
    
    BOOST_CHECK(x != 0.0f);
    BOOST_CHECK_EQUAL((int)x, 9);
    BOOST_CHECK_CLOSE(x, 9.5f, 0.0001f);
}

BOOST_AUTO_TEST_SUITE_END()
