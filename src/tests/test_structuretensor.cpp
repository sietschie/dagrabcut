#include <boost/test/unit_test.hpp>
#include "../lib/structuretensor.hpp"

struct Data2
{
    int value;
    cv::Mat randomImage;
    Data2()
    {
        BOOST_TEST_MESSAGE("set up data");
        value = 5;
        randomImage.create(128,128, CV_8UC3);
        cv::randu(randomImage, cv::Scalar(0), cv::Scalar(256));
    }

    ~Data2()
    {
        BOOST_TEST_MESSAGE("tear down data");
    }
};

BOOST_FIXTURE_TEST_SUITE(TestCaseST, Data2)

BOOST_AUTO_TEST_CASE(TestStructureTensor)
{
    cv::Vec3d v;
    v[0] = 1.0;
    v[1] = 2.0;
    v[2] = 3.0;

    StructureTensor st(v);

    cv::Mat stmat = st.getMatrix();
    BOOST_CHECK_EQUAL(v[0], stmat.at<double>(0,0));
    BOOST_CHECK_EQUAL(v[1], stmat.at<double>(1,1));
    BOOST_CHECK_EQUAL(v[2], stmat.at<double>(0,1));
    BOOST_CHECK_EQUAL(v[2], stmat.at<double>(1,0));

}


BOOST_AUTO_TEST_CASE(TestStructureTensorImage)
{
    StructureTensorImage sti(randomImage);

}

BOOST_AUTO_TEST_CASE(MyTestCaseSomething)
{
    int i = 5;
    int j = 10;
    BOOST_CHECK_EQUAL(i, value);

    
}

BOOST_AUTO_TEST_SUITE_END()
