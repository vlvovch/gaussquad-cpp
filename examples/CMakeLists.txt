add_executable(example_quadratures example_quadratures.cpp)
target_link_libraries(example_quadratures PRIVATE GaussQuad::GaussQuad)
set_property(TARGET example_quadratures PROPERTY FOLDER "gaussquad/examples")

add_executable(example_runge example_runge.cpp)
target_link_libraries(example_runge PRIVATE GaussQuad::GaussQuad)
set_property(TARGET example_runge PROPERTY FOLDER "gaussquad/examples")
