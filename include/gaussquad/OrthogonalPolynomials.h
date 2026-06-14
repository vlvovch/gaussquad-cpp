/**
 * @file OrthogonalPolynomials.h
 * @brief Header file defining orthogonal polynomial families for Gaussian quadrature.
 *
 * This file contains the abstract base class OrthogonalPolynomial and concrete
 * implementations for various families of orthogonal polynomials commonly used
 * in numerical integration via Gaussian quadrature.
 *
 * Each orthogonal polynomial family is characterized by:
 * - A three-term recurrence relation
 * - A weight function w(x)
 * - A normalization constant h_n
 * - An interval of orthogonality
 *
 * The recurrence relation has the form:
 * p_{n+1}(x) = (k_n x - a_n)p_n(x) - b_n p_{n-1}(x)
 * where k_n, a_n, b_n are the recurrence coefficients.
 */

#ifndef ORTHOGONAL_POLYNOMIALS_H
#define ORTHOGONAL_POLYNOMIALS_H
#include <vector>
#include <cmath>
#include <stdexcept>

namespace gaussquad {

  namespace detail {
    // Local pi constant. M_PI is not standard C++ and is unavailable on some
    // toolchains (e.g. MSVC without _USE_MATH_DEFINES), so it is not used here.
    constexpr double pi = 3.141592653589793238462643383279502884;
  }

  /**
   * @brief Abstract base class for orthogonal polynomials.
   *
   * This class defines the interface for orthogonal polynomial families.
   * Each derived class must implement the recurrence relation coefficients,
   * normalization constant, and weight function.
   *
   * The orthogonal polynomials satisfy the orthogonality condition:
   * ∫[a,b] w(x) p_m(x) p_n(x) dx = h_n δ_{mn}
   *
   * where [a,b] is the interval of orthogonality, w(x) is the weight function,
   * h_n is the normalization constant, and δ_{mn} is the Kronecker delta.
   */

  class OrthogonalPolynomial {
  public:
    /**
     * @brief Virtual destructor so derived instances can be safely deleted
     *        through a base-class pointer.
     */
    virtual ~OrthogonalPolynomial() = default;

    /**
     * @brief Get the recurrence relation coefficients for polynomial of order n.
     *
     * Returns the coefficients [k_n, a_n, b_n] for the three-term recurrence relation:
     * p_{n+1}(x) = (k_n x - a_n)p_n(x) - b_n p_{n-1}(x)
     *
     * @param n Order of the polynomial (n ≥ 0)
     * @return Vector containing [k_n, a_n, b_n] coefficients
     */
    virtual std::vector<double> Recurrence(int n) const = 0;

    /**
     * @brief Get the normalization constant for polynomial of order n.
     *
     * Returns h_n = ∫[a,b] w(x) p_n(x)² dx, the squared norm of the polynomial.
     * This is used in computing the quadrature weights.
     *
     * @param n Order of the polynomial (n ≥ 0)
     * @return Normalization constant h_n
     */
    virtual double Norm(int n) const = 0;

    /**
     * @brief Evaluate the weight function at point x.
     *
     * Returns w(x), the weight function associated with this polynomial family.
     * The weight function defines the measure for the inner product and integration.
     *
     * @param x Point at which to evaluate the weight function
     * @return Value of the weight function w(x)
     */
    virtual double Weight(double x) const = 0;
  };

  /**
   * @brief Implementation of Legendre polynomials.
   *
   * Legendre polynomials are orthogonal on the interval [-1, 1] with weight function w(x) = 1.
   * They are solutions to Legendre's differential equation and are widely used in
   * physics and engineering applications.
   *
   * Properties:
   * - Interval: [-1, 1]
   * - Weight function: w(x) = 1
   * - Recurrence: (n+1) P_{n+1}(x) = ((2n+1)x - 0)P_n(x) - n P_{n-1}(x)
   * - Normalization: h_n = 2/(2n+1)
   */
  class LegendrePolynomial : public OrthogonalPolynomial {
  public:
    std::vector<double> Recurrence(int n) const override {
      return { (2.*n + 1.)/(n+1.),
                0.,
                n/(n+1.) };
    }

    double Norm(int n) const override {
      return 2. / (2.*n + 1.);
    }

    double Weight(double /*x*/) const override {
      return 1.;
    }
  };

  /**
   * @brief Implementation of Laguerre polynomials.
   *
   * Laguerre polynomials are orthogonal on the interval [0, ∞) with weight function w(x) = exp(-x).
   * They are solutions to Laguerre's differential equation and are commonly used in
   * quantum mechanics and probability theory.
   *
   * Properties:
   * - Interval: [0, ∞)
   * - Weight function: w(x) = exp(-x)
   * - Recurrence: (n+1) L_{n+1}(x) = (-x + (2n+1))L_n(x) - n L_{n-1}(x)
   * - Normalization: h_n = 1
   */
  class LaguerrePolynomial : public OrthogonalPolynomial {
  public:
    virtual std::vector<double> Recurrence(int n) const override {
      return { -1. / (n + 1.),
                -(2. * n + 1) / (n + 1.),
                n / (n + 1.) };
    }

    virtual double Norm(int /*n*/) const override {
      return 1.;
    }

    virtual double Weight(double x) const override {
      return exp(-x);
    }
  };

  /**
   * @brief Implementation of generalized Laguerre polynomials.
   *
   * Generalized Laguerre polynomials L_n^(α)(x) are orthogonal on the interval [0, ∞)
   * with weight function w(x) = x^α exp(-x), where α > -1.
   * They generalize the standard Laguerre polynomials (α = 0).
   *
   * Properties:
   * - Interval: [0, ∞)
   * - Weight function: w(x) = x^α exp(-x)
   * - Parameter: α > -1
   * - Recurrence: (n+1) L_{n+1}^(α)(x) = (-x + (2n+1+α))L_n^(α)(x) - (n+α) L_{n-1}^(α)(x)
   * - Normalization: h_n = Γ(n+α+1)/Γ(n+1)
   */
  class GeneralizedLaguerrePolynomial : public LaguerrePolynomial {
    double m_alpha;
  public:
    GeneralizedLaguerrePolynomial(double alpha) : m_alpha(alpha) {
      if (alpha <= -1.)
        throw std::invalid_argument("GeneralizedLaguerrePolynomial: alpha must be > -1");
    }
    std::vector<double> Recurrence(int n) const override {
      return { -1. / (n + 1.),
               -(2. * n + 1 + m_alpha) / (n + 1.),
               (n + m_alpha) / (n + 1.) };
    }

    double Norm(int n) const override {
      return tgamma(n + m_alpha + 1.) / tgamma(n + 1.);
    }

    double Weight(double x) const override {
      return exp(-x) * pow(x, m_alpha);
    }
  };

  /**
   * @brief Implementation of Hermite polynomials.
   *
   * Hermite polynomials are orthogonal on the interval (-∞, ∞) with weight function w(x) = exp(-x²).
   * They are solutions to Hermite's differential equation and are fundamental in quantum mechanics,
   * particularly in the quantum harmonic oscillator.
   *
   * Properties:
   * - Interval: (-∞, ∞)
   * - Weight function: w(x) = exp(-x²)
   * - Recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
   * - Normalization: h_n = √π * 2^n * n!
   */
  class HermitePolynomial : public OrthogonalPolynomial {
  public:
    std::vector<double> Recurrence(int n) const override {
      return { 2.,
                0.,
                2. * n };
    }

    double Norm(int n) const override {
      double ret = sqrt(detail::pi);
      for (int i = 1; i <= n; i++) {
        ret *= 2. * i;
      }
      return ret;
    }

    double Weight(double x) const override {
      return exp(-x * x);
    }
  };

  /**
   * @brief Implementation of Jacobi polynomials.
   *
   * Jacobi polynomials P_n^(α,β)(x) are orthogonal on the interval [-1, 1] with weight function
   * w(x) = (1-x)^α (1+x)^β, where α, β > -1. They are the most general class of classical
   * orthogonal polynomials, with Legendre polynomials as a special case (α = β = 0).
   *
   * Properties:
   * - Interval: [-1, 1]
   * - Weight function: w(x) = (1-x)^α (1+x)^β
   * - Parameters: α, β > -1
   * - Recurrence: Complex three-term recurrence relation
   * - Normalization: h_n = 2^(α+β+1) * Γ(n+α+1) * Γ(n+β+1) / ((2n+α+β+1) * Γ(n+1) * Γ(n+α+β+1))
   */
  class JacobiPolynomial : public OrthogonalPolynomial {
    double m_alpha, m_beta;
  public:
    JacobiPolynomial(double alpha, double beta) : m_alpha(alpha), m_beta(beta) {
      if (alpha <= -1. || beta <= -1.)
        throw std::invalid_argument("JacobiPolynomial: alpha and beta must be > -1");
    }
    std::vector<double> Recurrence(int n) const override {
      // The general formula below has a removable 0/0 at n == 0 whenever
      // alpha + beta == 0 (e.g. Legendre (0,0)) or alpha + beta == -1 (e.g.
      // Chebyshev (-1/2,-1/2)). Use the closed form for P_1 directly:
      //   P_1(x) = (alpha+beta+2)/2 * x + (alpha-beta)/2
      // i.e. p_1 = (k_0 x - a_0) p_0 with k_0, a_0 below and b_0 = 0.
      if (n == 0) {
        return { 0.5 * (m_alpha + m_beta + 2.),   // k_0
                 -0.5 * (m_alpha - m_beta),         // a_0
                 0. };                              // b_0
      }
      double a = n + 1. + m_alpha;
      double b = n + 1. + m_beta;
      double c = a + b;
      double den = 2. * (n + 1.) * (c - n - 1.) * (c - 2);
      double an = (c - 1.) * c * (c - 2.) / den;
      double aln = -(a - b) * (c - 2. * n - 2.) / den;
      double betn = 2. * (a - 1.) * (b - 1.) * c / den;

      return { an,
               aln,
               betn };
    }

    double Norm(int n) const override {
      // h_0 = integral of the weight = 2^(a+b+1) * B(a+1,b+1)
      //     = 2^(a+b+1) * Gamma(a+1) Gamma(b+1) / Gamma(a+b+2).
      // The general-n formula below has a (2n+a+b+1) denominator and a
      // Gamma(n+a+b+1) factor that are singular at n == 0 when a+b == -1;
      // the closed form for h_0 avoids both.
      if (n == 0) {
        return pow(2., m_alpha + m_beta + 1.)
          * tgamma(m_alpha + 1.) * tgamma(m_beta + 1.)
          / tgamma(m_alpha + m_beta + 2.);
      }
      return pow(2., m_alpha + m_beta + 1.) / (2. * n + m_alpha + m_beta + 1.)
      * tgamma(n + m_alpha + 1.) / tgamma(n + 1.)
      * tgamma(n + m_beta + 1.) / tgamma(n + m_alpha + m_beta + 1.);
    }

    double Weight(double x) const override {
      return pow(1. - x, m_alpha) * pow(1. + x, m_beta);
    }
  };
}

#endif