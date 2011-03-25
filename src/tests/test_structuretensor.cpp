#include <boost/test/unit_test.hpp>
#include "../lib/structuretensor.hpp"

struct Data2
{
    int value;
    Data2()
    {
        BOOST_TEST_MESSAGE("set up data");
        value = 5;
    }

    ~Data2()
    {
        BOOST_TEST_MESSAGE("tear down data");
    }
};

BOOST_FIXTURE_TEST_SUITE(TestCaseST, Data2)

BOOST_AUTO_TEST_CASE(MyTestCaseSomething)
{
    int i = 5;
    int j = 10;
    BOOST_CHECK_EQUAL(i, value);
}

BOOST_AUTO_TEST_SUITE_END()
