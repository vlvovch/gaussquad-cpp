/**
 * @file gaussquad.h
 * @brief Umbrella header for the GaussQuad library.
 *
 * Single include that pulls in the orthogonal polynomial families and the
 * Gaussian quadrature node/weight computation. Everything lives in the
 * `gaussquad` namespace.
 *
 *     #include <gaussquad/gaussquad.h>
 *     gaussquad::LegendrePolynomial leg;
 *     auto rule = gaussquad::GaussianQuadrature(64, &leg);
 */
#ifndef GAUSSQUAD_GAUSSQUAD_H
#define GAUSSQUAD_GAUSSQUAD_H

#include "gaussquad/OrthogonalPolynomials.h"
#include "gaussquad/GaussianQuadratures.h"

#endif
