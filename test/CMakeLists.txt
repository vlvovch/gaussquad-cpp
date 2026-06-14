# Link against gtest_main and the header-only library under test.
add_executable(test_GaussianQuadratures test_GaussianQuadratures.cpp)
target_link_libraries(test_GaussianQuadratures GaussQuad::GaussQuad GTest::gtest_main)
set_property(TARGET test_GaussianQuadratures PROPERTY FOLDER tests)

include(GoogleTest)
gtest_discover_tests(test_GaussianQuadratures)
