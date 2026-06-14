/*
 * GaussQuad library
 *
 * Copyright (c) 2025 Volodymyr Vovchenko
 *
 * Licensed under the MIT License <http://opensource.org/licenses/MIT>
 */
#include "gtest/gtest.h"
#include "gaussquad/gaussquad.h"

#include <cmath>
#include <vector>
#include <algorithm>

using namespace gaussquad;

namespace {

  // Local pi constant (M_PI is not standard C++ and may be absent on MSVC).
  constexpr double pi = 3.141592653589793238462643383279502884;

  // Round-off floor for the sum-of-weights check grows with the number of nodes
  // (summing n weights accumulates ~ n * machine_epsilon).
  double sumTol(int n) { return 1e-13 * std::max(1, n); }

  double sumOfWeights(const std::pair<std::vector<double>, std::vector<double>>& q) {
    double s = 0.0;
    for (double w : q.second) s += w;
    return s;
  }

  // --- Sum rule: sum of weights == integral of the weight function ---------

  TEST(SumRule, Legendre) {              // w(x) = 1 on [-1,1] -> 2
    LegendrePolynomial p;
    for (int n = 1; n <= 1000; n += 10)
      EXPECT_NEAR(sumOfWeights(GaussianQuadrature(n, &p)), 2.0, sumTol(n)) << "n=" << n;
  }

  TEST(SumRule, Laguerre) {              // w(x) = e^-x on [0,inf) -> 1
    LaguerrePolynomial p;
    for (int n = 1; n <= 1000; n += 10)
      EXPECT_NEAR(sumOfWeights(GaussianQuadrature(n, &p)), 1.0, sumTol(n)) << "n=" << n;
  }

  TEST(SumRule, GeneralizedLaguerre) {   // w(x) = x^a e^-x -> Gamma(a+1)
    const double alpha = -0.5;
    GeneralizedLaguerrePolynomial p(alpha);
    const double expected = std::tgamma(alpha + 1.0);
    for (int n = 1; n <= 1000; n += 10)
      EXPECT_NEAR(sumOfWeights(GaussianQuadrature(n, &p)), expected, sumTol(n)) << "n=" << n;
  }

  TEST(SumRule, Hermite) {               // w(x) = e^-x^2 -> sqrt(pi)
    HermitePolynomial p;
    const double expected = std::sqrt(pi);
    for (int n = 1; n <= 1000; n += 10)
      EXPECT_NEAR(sumOfWeights(GaussianQuadrature(n, &p)), expected, sumTol(n)) << "n=" << n;
  }

  TEST(SumRule, Jacobi) {                // w(x) = (1-x)^a (1+x)^b -> 2^(a+b+1) B(a+1,b+1)
    const double a = 0.5, b = 0.75;
    JacobiPolynomial p(a, b);
    const double expected = std::pow(2.0, a + b + 1.0) * std::tgamma(a + 1.0) * std::tgamma(b + 1.0)
                          / std::tgamma(a + b + 2.0);
    for (int n = 1; n <= 1000; n += 10)
      EXPECT_NEAR(sumOfWeights(GaussianQuadrature(n, &p)), expected, sumTol(n)) << "n=" << n;
  }

  TEST(SumRule, LegendreNewtonPath) {    // GaussianQuadraturesLegendre (asymptotic + Newton)
    for (int n = 1; n <= 1000; n += 10)
      EXPECT_NEAR(sumOfWeights(GaussianQuadraturesLegendre(n)), 2.0, sumTol(n)) << "n=" << n;
  }

  // --- Integration accuracy on known integrals -----------------------------

  TEST(Integral, LegendreRunge) {        // \int_{-1}^1 dx/(1+25x^2) = (2/5) atan 5
    LegendrePolynomial p;
    const double exact = (2.0 / 5.0) * std::atan(5.0);
    auto q = GaussianQuadrature(200, &p);
    double I = 0.0;
    for (size_t k = 0; k < q.first.size(); ++k)
      I += q.second[k] / (1.0 + 25.0 * q.first[k] * q.first[k]);
    EXPECT_NEAR(I, exact, 1e-12);
  }

  TEST(Integral, LaguerreCosine) {       // \int_0^inf e^-x cos x dx = 1/2 (weight folds in e^-x)
    LaguerrePolynomial p;
    auto q = GaussianQuadrature(64, &p);
    double I = 0.0;
    for (size_t k = 0; k < q.first.size(); ++k) I += q.second[k] * std::cos(q.first[k]);
    EXPECT_NEAR(I, 0.5, 1e-12);
  }

  TEST(Integral, HermiteSecondMoment) {  // \int e^-x^2 x^2 dx = sqrt(pi)/2
    HermitePolynomial p;
    auto q = GaussianQuadrature(64, &p);
    double I = 0.0;
    for (size_t k = 0; k < q.first.size(); ++k) I += q.second[k] * q.first[k] * q.first[k];
    EXPECT_NEAR(I, std::sqrt(pi) / 2.0, 1e-12);
  }

  // --- Structural properties ----------------------------------------------

  TEST(Properties, NodesAscendingAndWeightsNonNegative) {
    // The classical families have distinct, ascending nodes and (mathematically)
    // strictly positive weights. In floating point the weights of the extreme
    // nodes of high-order Laguerre/Hermite rules can underflow to exactly 0
    // (their true value is below ~1e-308); a *negative* weight, however, would
    // signal a genuine bug, so we assert non-negativity.
    LegendrePolynomial leg; LaguerrePolynomial lag; HermitePolynomial her;
    const OrthogonalPolynomial* polys[] = { &leg, &lag, &her };
    for (const OrthogonalPolynomial* p : polys) {
      for (int n : {8, 64, 256}) {
        auto q = GaussianQuadrature(n, p);
        for (int k = 1; k < n; ++k)
          EXPECT_LT(q.first[k - 1], q.first[k]) << "nodes not ascending at k=" << k;
        for (int k = 0; k < n; ++k)
          EXPECT_GE(q.second[k], 0.0) << "negative weight at k=" << k;
      }
    }
  }

  TEST(Properties, HighOrderDoesNotBreakDown) {
    // High-order Laguerre/Hermite used to overflow/diverge with the old
    // Newton-Raphson weight formula; the QL Golub-Welsch path stays finite.
    LaguerrePolynomial lag; HermitePolynomial her;
    for (const OrthogonalPolynomial* p : { static_cast<const OrthogonalPolynomial*>(&lag),
                                           static_cast<const OrthogonalPolynomial*>(&her) }) {
      auto q = GaussianQuadrature(1000, p);
      for (double x : q.first)  EXPECT_TRUE(std::isfinite(x));
      for (double w : q.second) EXPECT_TRUE(std::isfinite(w));
    }
  }

  TEST(Properties, NodesGolubWelschMatchesFull) {
    LegendrePolynomial p;
    auto full = GaussianQuadrature(128, &p);
    auto nodes = NodesGolubWelsch(128, &p);
    ASSERT_EQ(nodes.size(), full.first.size());
    for (size_t k = 0; k < nodes.size(); ++k) EXPECT_DOUBLE_EQ(nodes[k], full.first[k]);
  }

  TEST(Validation, NonPositiveNodeCountThrows) {
    // n == 0 would index empty vectors; n < 0 would resize() to a huge size_t.
    LegendrePolynomial leg;
    for (int n : {0, -1, -100}) {
      EXPECT_THROW(GaussianQuadrature(n, &leg),        std::invalid_argument) << "n=" << n;
      EXPECT_THROW(GolubWelschNodesWeights(n, &leg),   std::invalid_argument) << "n=" << n;
      EXPECT_THROW(NodesGolubWelsch(n, &leg),          std::invalid_argument) << "n=" << n;
      EXPECT_THROW(GaussianQuadraturesLegendre(n),     std::invalid_argument) << "n=" << n;
      EXPECT_THROW(InitialGuessLegendre(n),            std::invalid_argument) << "n=" << n;
      EXPECT_THROW(InitialGuessLaguerre(n),            std::invalid_argument) << "n=" << n;
      EXPECT_THROW(InitialGuessHermite(n),             std::invalid_argument) << "n=" << n;
    }
  }

  // --- Jacobi edge cases (removable singularities at n == 0) ---------------

  TEST(JacobiEdgeCases, ReducesToLegendre) {
    // Jacobi(0,0) must reproduce Legendre nodes and weights exactly (alpha+beta==0).
    LegendrePolynomial leg;
    JacobiPolynomial jac00(0.0, 0.0);
    for (int n : {1, 8, 32, 64}) {
      auto L = GaussianQuadrature(n, &leg);
      auto J = GaussianQuadrature(n, &jac00);
      ASSERT_EQ(J.first.size(), L.first.size());
      for (int k = 0; k < n; ++k) {
        EXPECT_NEAR(J.first[k],  L.first[k],  1e-12) << "node n=" << n << " k=" << k;
        EXPECT_NEAR(J.second[k], L.second[k], 1e-12) << "weight n=" << n << " k=" << k;
      }
    }
  }

  TEST(JacobiEdgeCases, AlphaPlusBetaZero) {
    // (0.5, -0.5): alpha+beta == 0 -> Recurrence(0) singularity. mu0 = pi.
    JacobiPolynomial jac(0.5, -0.5);
    for (int n : {1, 16, 64}) {
      auto q = GaussianQuadrature(n, &jac);
      double s = 0.0;
      for (double x : q.first)  EXPECT_TRUE(std::isfinite(x)) << "n=" << n;
      for (double w : q.second) { EXPECT_TRUE(std::isfinite(w)) << "n=" << n; s += w; }
      EXPECT_NEAR(s, pi, 1e-12) << "n=" << n;   // integral of (1-x)^0.5 (1+x)^-0.5 = pi
    }
  }

  TEST(JacobiEdgeCases, AlphaPlusBetaMinusOneChebyshev) {
    // (-0.5, -0.5): alpha+beta == -1 (Chebyshev 1st kind). Exercises BOTH the
    // Recurrence(0) and Norm(0) special cases. mu0 = pi.
    JacobiPolynomial jac(-0.5, -0.5);
    for (int n : {1, 16, 64}) {
      auto q = GaussianQuadrature(n, &jac);
      double s = 0.0;
      for (double x : q.first)  EXPECT_TRUE(std::isfinite(x)) << "n=" << n;
      for (double w : q.second) { EXPECT_TRUE(std::isfinite(w)) << "n=" << n; s += w; }
      EXPECT_NEAR(s, pi, 1e-12) << "n=" << n;   // integral of 1/sqrt(1-x^2) = pi
    }
  }

  // --- Parameter-domain validation ----------------------------------------

  TEST(ParameterValidation, InvalidParametersThrow) {
    EXPECT_THROW(GeneralizedLaguerrePolynomial(-1.0), std::invalid_argument);
    EXPECT_THROW(GeneralizedLaguerrePolynomial(-1.5), std::invalid_argument);
    EXPECT_NO_THROW(GeneralizedLaguerrePolynomial(-0.5));

    EXPECT_THROW(JacobiPolynomial(-1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(JacobiPolynomial(0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(JacobiPolynomial(-2.0, 0.5), std::invalid_argument);
    EXPECT_NO_THROW(JacobiPolynomial(-0.5, -0.5));   // valid: Chebyshev 1st kind
  }

}
