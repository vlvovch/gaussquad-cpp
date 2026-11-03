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
#include <limits>
#include <string>

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

  // --- Weighted-moment accuracy (catches node errors, not just weight sums) -

  // B(p,q) = Gamma(p) Gamma(q) / Gamma(p+q), evaluated in log space.
  double beta(double p, double q) {
    return std::exp(std::lgamma(p) + std::lgamma(q) - std::lgamma(p + q));
  }

  // Exact m-th moment of the Jacobi weight:
  //   \int_{-1}^1 (1-x)^a (1+x)^b x^m dx
  // Substitute x = 2t-1 (t in [0,1]); expand x^m = sum_k C(m,k) (2t)^k (-1)^(m-k):
  //   = 2^(a+b+1) sum_k C(m,k) 2^k (-1)^(m-k) B(b+k+1, a+1).
  // This is an independent closed form -- it never touches the quadrature code,
  // so it pins down node placement (unlike the moment-0 sum-of-weights check).
  double jacobiMomentExact(int m, double a, double b) {
    double s = 0.0, binom = 1.0;              // binom = C(m,k)
    for (int k = 0; k <= m; ++k) {
      double sign = ((m - k) % 2 == 0) ? 1.0 : -1.0;
      s += binom * std::pow(2.0, k) * sign * beta(b + k + 1.0, a + 1.0);
      binom = binom * (m - k) / (k + 1.0);    // C(m,k) -> C(m,k+1)
    }
    return std::pow(2.0, a + b + 1.0) * s;
  }

  double weightedMoment(const std::pair<std::vector<double>, std::vector<double>>& q, int m) {
    double I = 0.0;
    for (size_t k = 0; k < q.first.size(); ++k) I += q.second[k] * std::pow(q.first[k], m);
    return I;
  }

  TEST(Moments, JacobiAsymmetric) {
    // alpha != beta activates the a_n recurrence coefficient. An n-point rule is
    // exact for polynomials up to degree 2n-1, so all these moments must match.
    const double a = 0.5, b = 0.75;
    JacobiPolynomial p(a, b);
    for (int n : {4, 8, 16}) {
      auto q = GaussianQuadrature(n, &p);
      for (int m = 0; m <= 6; ++m)
        EXPECT_NEAR(weightedMoment(q, m), jacobiMomentExact(m, a, b), 1e-12)
          << "n=" << n << " m=" << m;
    }
  }

  TEST(Moments, JacobiAsymmetricNegativeParams) {
    const double a = -0.25, b = 0.6;
    JacobiPolynomial p(a, b);
    for (int n : {5, 12}) {
      auto q = GaussianQuadrature(n, &p);
      for (int m = 0; m <= 6; ++m)
        EXPECT_NEAR(weightedMoment(q, m), jacobiMomentExact(m, a, b), 1e-12)
          << "n=" << n << " m=" << m;
    }
  }

  TEST(Moments, GeneralizedLaguerre) {
    // \int_0^inf x^alpha e^-x x^m dx = Gamma(alpha + m + 1).
    const double alpha = 1.5;
    GeneralizedLaguerrePolynomial p(alpha);
    for (int n : {4, 8, 16}) {
      auto q = GaussianQuadrature(n, &p);
      for (int m = 0; m <= 6; ++m)
        EXPECT_NEAR(weightedMoment(q, m), std::tgamma(alpha + m + 1.0), 1e-9)
          << "n=" << n << " m=" << m;
    }
  }

  TEST(Norm, GeneralizedLaguerreHighOrderIsFinite) {
    // Golub-Welsch reads only Norm(0), so the rule-based tests above cannot see
    // an overflow in Norm(n) at large n, which the Newton weight formula uses.
    // h_n = Gamma(n+alpha+1)/Gamma(n+1), which is n + 1 for alpha = 1; the old
    // tgamma ratio gave inf at n = 170 and inf/inf = NaN from n = 171 on.
    GeneralizedLaguerrePolynomial p(1.0);
    for (int n : {170, 200, 500}) {
      const double h = p.Norm(n);
      ASSERT_TRUE(std::isfinite(h)) << "n=" << n;
      EXPECT_NEAR(h / (n + 1.0), 1.0, 1e-10) << "n=" << n;
    }
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

  // Run the Newton solver on a 2-point Legendre rule from the seed x0 and expect
  // the derivative guard, specifically, to throw. Without the guard these seeds
  // return NaN nodes and weights rather than throwing at all.
  void ExpectNewtonDerivativeGuardThrows(const std::vector<double>& x0) {
    LegendrePolynomial leg;
    try {
      GaussianQuadraturesNewton(2, x0, &leg);
      ADD_FAILURE() << "expected std::runtime_error from the derivative guard";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find("derivative"), std::string::npos) << e.what();
    }
  }

  TEST(Validation, NewtonThrowsOnZeroOrNonFiniteDerivative) {
    // P_2'(0) == 0 exactly, so the first Newton step would divide by zero.
    ExpectNewtonDerivativeGuardThrows({0.0, 0.0});
    // A non-finite seed gives a non-finite derivative.
    ExpectNewtonDerivativeGuardThrows({std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN()});
    ExpectNewtonDerivativeGuardThrows({std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::infinity()});
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

  // --- Large weight exponents: h_n where tgamma overflows ------------------

  // Integral of the Jacobi weight for integer beta, computed without any gamma
  // function: mu0(a, 0) = 2^(a+1)/(a+1) and mu0(a, b+1) = mu0(a, b) * 2(b+1)/(a+b+2).
  // Every factor is O(1), so this is well conditioned and independent of the
  // library's tgamma/lgamma code.
  double JacobiMu0IntegerBeta(double a, int b) {
    double mu = std::pow(2.0, a + 1.0) / (a + 1.0);
    for (int k = 0; k < b; ++k) mu *= 2.0 * (k + 1) / (a + k + 2.0);
    return mu;
  }

  TEST(LargeParameters, JacobiRuleIsFiniteAndExact) {
    // With exponents this large the tgamma products in Norm(0) overflow
    // (inf/inf = NaN), which used to turn every weight into NaN.
    const double params[][2] = { {100., 100.}, {250., 250.}, {300.5, 2.}, {2., 300.} };
    for (const auto& ab : params) {
      const double a = ab[0], b = ab[1];
      JacobiPolynomial jac(a, b);
      const double mu0 = JacobiMu0IntegerBeta(a, static_cast<int>(b));
      // Normalised moments m_k = E[x^k] under the weight, from the gamma-free
      // recurrence obtained by integrating d/dx[x^k (1-x)^(a+1) (1+x)^(b+1)]:
      //   (a+b+k+2) m_{k+1} = (b-a) m_k + k m_{k-1},   m_0 = 1.
      // Both terms on the right have the same sign, so there is no cancellation.
      // Moments beyond the second depend on the recurrence coefficients for
      // n >= 1 (the first two only on the separate n == 0 branch), so they also
      // check the a_n coefficient that was wrong for alpha != beta.
      const int maxMoment = 6;
      std::vector<double> m(maxMoment + 1);
      m[0] = 1.;
      m[1] = (b - a) / (a + b + 2.);
      for (int k = 1; k < maxMoment; ++k)
        m[k + 1] = ((b - a) * m[k] + k * m[k - 1]) / (a + b + k + 2.);
      for (int n : {2, 8, 32}) {
        auto q = GaussianQuadrature(n, &jac);
        std::vector<double> s(maxMoment + 1, 0.);
        for (int k = 0; k < n; ++k) {
          const double x = q.first[k], w = q.second[k];
          ASSERT_TRUE(std::isfinite(x)) << "a=" << a << " b=" << b << " n=" << n;
          ASSERT_TRUE(std::isfinite(w)) << "a=" << a << " b=" << b << " n=" << n;
          EXPECT_GE(w, 0.0);
          double xp = 1.;
          for (int j = 0; j <= maxMoment; ++j) { s[j] += w * xp; xp *= x; }
        }
        EXPECT_NEAR(s[0] / mu0, 1.0, 1e-10) << "a=" << a << " b=" << b << " n=" << n;
        // An n-point Gauss rule is exact to degree 2n - 1.
        for (int j = 1; j <= std::min(maxMoment, 2 * n - 1); ++j)
          EXPECT_NEAR(s[j] / s[0], m[j], 1e-12)
              << "a=" << a << " b=" << b << " n=" << n << " moment " << j;
      }
    }
  }

  TEST(LargeParameters, NormMatchesRatioRecurrence) {
    // h_n / h_{n-1} has a closed form made of O(1) factors. Check Norm(n)
    // against it on both sides of the point where tgamma overflows.

    // Generalized Laguerre: h_n / h_{n-1} = (n + alpha) / n.
    // tgamma(n + 151) overflows from n = 21 on.
    const double alpha = 150.;
    GeneralizedLaguerrePolynomial lag(alpha);
    for (int n = 1; n <= 30; ++n) {
      ASSERT_TRUE(std::isfinite(lag.Norm(n))) << "n=" << n;
      EXPECT_NEAR(lag.Norm(n) / (lag.Norm(n - 1) * (n + alpha) / n), 1.0, 1e-10) << "n=" << n;
    }

    // Jacobi: h_n / h_{n-1} = (2n+a+b-1)(n+a)(n+b) / ((2n+a+b+1) n (n+a+b)).
    const double params[][2] = { {100., 100.}, {300.5, 2.} };
    for (const auto& ab : params) {
      const double a = ab[0], b = ab[1];
      JacobiPolynomial jac(a, b);
      EXPECT_NEAR(jac.Norm(0) / JacobiMu0IntegerBeta(a, static_cast<int>(b)), 1.0, 1e-10)
          << "a=" << a << " b=" << b;
      for (int n = 1; n <= 30; ++n) {
        const double s = 2. * n + a + b;
        const double ratio = (s - 1.) * (n + a) * (n + b) / ((s + 1.) * n * (n + a + b));
        ASSERT_TRUE(std::isfinite(jac.Norm(n))) << "a=" << a << " b=" << b << " n=" << n;
        EXPECT_NEAR(jac.Norm(n) / (jac.Norm(n - 1) * ratio), 1.0, 1e-10)
            << "a=" << a << " b=" << b << " n=" << n;
      }
    }
  }

  // Tolerance for a computed mu0 against an exact reference: 1e-14, or a few
  // ulps of log(mu0) where mu0 is so large or small that exp() of a computed
  // logarithm cannot do better.
  double Mu0Tolerance(double ref) {
    return std::max(1e-14, 3. * std::numeric_limits<double>::epsilon() * std::fabs(std::log(ref)));
  }

  TEST(LargeParameters, JacobiWeightIntegralIsAccurate) {
    // mu0 = Norm(0) scales every Golub-Welsch weight. A plain sum of lgamma
    // values loses accuracy as the exponents grow (3e-14 relative at
    // alpha = beta = 100, 2e-12 at 1000, 7e-3 at 1e12), because its large
    // terms cancel almost completely; the library cancels them analytically.

    // alpha = beta: by the duplication formula, mu0 = sqrt(pi) Gamma(x) / Gamma(x + 1/2)
    // with x = alpha + 1, and for large x
    //   Gamma(x + 1/2) / Gamma(x) = sqrt(x) (1 - 1/(8x) + 1/(128x^2) + 5/(1024x^3) - 21/(32768x^4) + O(x^-5)),
    // which for x >= 1e4 is accurate to ~1e-23 relative. The largest cases have
    // alpha + beta beyond the double range although mu0 ~ 1e-154 is not; the
    // square roots are taken separately so that pi / x does not underflow.
    for (double a : {1e4, 1e6, 1e9, 1e12, 1e200, 1e308, std::numeric_limits<double>::max()}) {
      const double x = a + 1.;
      const double ref = std::sqrt(pi) / std::sqrt(x)
          / (1. - 1. / (8. * x) + 1. / (128. * x * x) + 5. / (1024. * x * x * x) - 21. / (32768. * x * x * x * x));
      EXPECT_NEAR(JacobiPolynomial(a, a).Norm(0) / ref, 1.0, Mu0Tolerance(ref)) << "alpha=beta=" << a;
    }

    // Where the true mu0 exceeds the double range, the result is +inf, not
    // NaN or 0 from an intermediate overflow such as s * s. Both argument
    // orders are checked.
    const double overflowing[][2] = { {1e200, 1e199}, {0.5, 1e300}, {20., 1e160}, {1e308, 1.0000000001e308},
                                      {20., 1e308}, {9.5, 1.7e308}, {1e10, 1e308} };
    for (const auto& ab : overflowing) {
      EXPECT_EQ(JacobiPolynomial(ab[0], ab[1]).Norm(0), std::numeric_limits<double>::infinity())
          << "alpha=" << ab[0] << " beta=" << ab[1];
      EXPECT_EQ(JacobiPolynomial(ab[1], ab[0]).Norm(0), std::numeric_limits<double>::infinity())
          << "alpha=" << ab[1] << " beta=" << ab[0];
    }

    // Integer beta: the gamma-free product reference.
    const double integerBeta[][2] = { {100., 100.}, {300., 300.}, {1000., 1000.}, {1000.5, 7.} };
    for (const auto& ab : integerBeta) {
      const double ref = JacobiMu0IntegerBeta(ab[0], static_cast<int>(ab[1]));
      EXPECT_NEAR(JacobiPolynomial(ab[0], ab[1]).Norm(0) / ref, 1.0, Mu0Tolerance(ref))
          << "alpha=" << ab[0] << " beta=" << ab[1];
    }

    // Non-integer and unequal exponents, against values computed with mpmath
    // at 60 digits for the double nearest each decimal literal. These cover
    // each branch: raising the smaller argument to 10 first ((-0.5, 30.25),
    // (-0.9, 61.8)), |alpha - beta| / (alpha + beta + 2) >= 1/2 ((9.5, 40.3)),
    // and < 1/2 but non-zero ((150.5, 220.25), (1e9, 1.0001e9)).
    const double tabulated[][3] = {
      { -0.5,  30.25,    574856433.33957649 },
      { -0.9,  61.8,     2.7077795187573254e+19 },
      {  9.5,  40.3,     7808.6417682065375 },
      { 150.5, 220.25,   93.835390843140222 },
      {  1e9,  1.0001e9, 0.00068272530425484414 },
    };
    for (const auto& t : tabulated) {
      EXPECT_NEAR(JacobiPolynomial(t[0], t[1]).Norm(0) / t[2], 1.0, Mu0Tolerance(t[2]))
          << "alpha=" << t[0] << " beta=" << t[1];
      // mu0 is symmetric in (alpha, beta).
      EXPECT_NEAR(JacobiPolynomial(t[1], t[0]).Norm(0) / t[2], 1.0, Mu0Tolerance(t[2]))
          << "alpha=" << t[1] << " beta=" << t[0];
    }
  }

  TEST(LargeParameters, JacobiWeightIntegralIsPositiveOrInfinite) {
    // Over every pair from a grid spanning the whole valid range (-1, DBL_MAX],
    // mu0 must be positive and either finite or +inf: never NaN or 0, which an
    // intermediate overflow or underflow would produce. mu0 is symmetric in
    // (alpha, beta); swapping them may change the last bit when both are
    // small, where the tgamma product is taken in argument order.
    const double grid[] = { -1. + 1e-12, -0.999, -0.5, 0., 0.5, 8.99, 9., 9.01, 10., 20., 100., 170., 1e3,
                            1e6, 1e12, 1e50, 1e150, 1e154, 1e155, 1e200, 1e300, 1e307, 9e307, 1e308, 1.7e308,
                            std::numeric_limits<double>::max() };
    for (double a : grid) {
      for (double b : grid) {
        const double mu0 = JacobiPolynomial(a, b).Norm(0);
        EXPECT_FALSE(std::isnan(mu0)) << "alpha=" << a << " beta=" << b;
        EXPECT_GT(mu0, 0.0) << "alpha=" << a << " beta=" << b;
        const double swapped = JacobiPolynomial(b, a).Norm(0);
        if (std::isinf(mu0))
          EXPECT_EQ(swapped, mu0) << "alpha=" << a << " beta=" << b;
        else
          EXPECT_NEAR(swapped / mu0, 1.0, 1e-15) << "alpha=" << a << " beta=" << b;
      }
    }
  }

  TEST(LargeParameters, NewtonWeightsWithLargeJacobiExponents) {
    // GaussianQuadraturesNewton forms its weights from Norm(n-1), which used
    // to be NaN here even with the exact nodes as the seed.
    JacobiPolynomial jac(100., 100.);
    const int n = 16;
    auto gw = GaussianQuadrature(n, &jac);
    auto nr = GaussianQuadraturesNewton(n, gw.first, &jac);
    for (int k = 0; k < n; ++k) {
      EXPECT_NEAR(nr.first[k], gw.first[k], 1e-14) << "k=" << k;
      EXPECT_NEAR(nr.second[k] / gw.second[k], 1.0, 1e-10) << "k=" << k;
    }
  }

  // --- Exact symmetry for even weight functions ----------------------------

  // A Legendre family that does not override HasEvenWeight(), like a
  // user-defined family would not: its rule is left as the eigensolver
  // returns it.
  class PlainLegendre : public OrthogonalPolynomial {
  public:
    std::vector<double> Recurrence(int n) const override {
      return { (2. * n + 1.) / (n + 1.), 0., n / (n + 1.) };
    }
    double Norm(int n) const override { return 2. / (2. * n + 1.); }
    double Weight(double /*x*/) const override { return 1.; }
  };

  void ExpectExactlySymmetric(const std::pair<std::vector<double>, std::vector<double>>& q) {
    const int n = static_cast<int>(q.first.size());
    for (int k = 0; k < n; ++k) {
      EXPECT_EQ(q.first[k], -q.first[n - 1 - k]) << "node k=" << k << " n=" << n;
      EXPECT_EQ(q.second[k], q.second[n - 1 - k]) << "weight k=" << k << " n=" << n;
    }
    if (n % 2 == 1) {
      EXPECT_EQ(q.first[n / 2], 0.0) << "middle node, n=" << n;
    }
  }

  TEST(Symmetry, HasEvenWeightFlags) {
    EXPECT_TRUE(LegendrePolynomial().HasEvenWeight());
    EXPECT_TRUE(HermitePolynomial().HasEvenWeight());
    EXPECT_TRUE(JacobiPolynomial(0.3, 0.3).HasEvenWeight());
    EXPECT_FALSE(JacobiPolynomial(0.3, 0.5).HasEvenWeight());
    EXPECT_FALSE(LaguerrePolynomial().HasEvenWeight());
    EXPECT_FALSE(GeneralizedLaguerrePolynomial(0.5).HasEvenWeight());
    EXPECT_FALSE(PlainLegendre().HasEvenWeight());   // the default for user-defined families
  }

  TEST(Symmetry, EvenWeightRulesAreExactlySymmetric) {
    LegendrePolynomial leg;
    HermitePolynomial her;
    JacobiPolynomial gegenbauer(0.5, 0.5), chebyshev(-0.5, -0.5), wide(100., 100.);
    const OrthogonalPolynomial* polys[] = { &leg, &her, &gegenbauer, &chebyshev, &wide };
    for (const OrthogonalPolynomial* p : polys)
      for (int n : {1, 2, 3, 10, 101, 1000})
        ExpectExactlySymmetric(GaussianQuadrature(n, p));
  }

  TEST(Symmetry, SymmetrizedRuleAgreesWithUnsymmetrized) {
    // Symmetrizing only moves nodes and weights at the round-off level
    // (measured: <= 1e-15 and <= 3e-15 up to n = 1000), and the result still
    // agrees with the independent Newton-Raphson Legendre path. Tolerances are
    // absolute: the Golub-Welsch weights carry an absolute error ~ eps * mu0,
    // so a relative comparison would be dominated by the tiny endpoint weights.
    LegendrePolynomial leg;
    PlainLegendre plain;
    for (int n : {10, 101, 1000}) {
      auto sym = GaussianQuadrature(n, &leg);
      auto raw = GaussianQuadrature(n, &plain);
      auto newton = GaussianQuadraturesLegendre(n);
      for (int k = 0; k < n; ++k) {
        EXPECT_NEAR(sym.first[k],  raw.first[k],     1e-14) << "n=" << n << " k=" << k;
        EXPECT_NEAR(sym.second[k], raw.second[k],    1e-14) << "n=" << n << " k=" << k;
        EXPECT_NEAR(sym.first[k],  newton.first[k],  1e-14) << "n=" << n << " k=" << k;
        EXPECT_NEAR(sym.second[k], newton.second[k], 1e-12) << "n=" << n << " k=" << k;
      }
    }
  }

}
