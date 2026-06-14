/**
 * @file example_runge.cpp
 * @brief Integrating the Runge function with Gauss-Legendre quadrature.
 *
 * The Runge function
 *
 *     f(x) = 1 / (1 + 25 x^2)
 *
 * is the classic counter-example of Runge's phenomenon: interpolating it on
 * equally spaced nodes (and integrating via the resulting Newton-Cotes rules)
 * *diverges* as the order grows. Gauss-Legendre quadrature, whose nodes cluster
 * towards the interval ends, instead converges rapidly. This example integrates
 *
 *     I = \int_{-1}^{1} dx / (1 + 25 x^2) = (2/5) * atan(5) = 0.549360306778...
 *
 * with increasing node counts and prints the error, showing the convergence.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "gaussquad/gaussquad.h"

using namespace std;
using namespace gaussquad;

// The Runge function.
static double runge(double x) {
  return 1.0 / (1.0 + 25.0 * x * x);
}

// Integrate f over [a, b] with an n-point Gauss-Legendre rule.
// The library returns the rule on [-1, 1]; we map it linearly to [a, b].
static double integrateLegendre(double (*f)(double), double a, double b, int n) {
  LegendrePolynomial poly;
  auto rule = GaussianQuadrature(n, &poly);
  const vector<double>& x = rule.first;   // nodes on [-1, 1]
  const vector<double>& w = rule.second;  // weights on [-1, 1]
  const double half = 0.5 * (b - a);
  const double mid  = 0.5 * (b + a);
  double sum = 0.0;
  for (size_t k = 0; k < x.size(); ++k)
    sum += w[k] * f(half * x[k] + mid);
  return half * sum;
}

int main() {
  const double exact = (2.0 / 5.0) * atan(5.0);

  cout << "Gauss-Legendre integration of the Runge function 1/(1+25 x^2) on [-1, 1]\n";
  cout << "Exact value: I = (2/5) atan(5) = " << setprecision(15) << exact << "\n\n";

  const int tab = 18;
  cout << setw(6) << "n" << setw(tab) << "approximation" << setw(tab) << "abs. error" << "\n";
  cout << scientific << setprecision(6);
  for (int n : {4, 8, 16, 24, 32, 48, 64, 128}) {
    double approx = integrateLegendre(runge, -1.0, 1.0, n);
    cout << fixed << setw(6) << n
         << scientific << setw(tab) << approx
         << setw(tab) << fabs(approx - exact) << "\n";
  }

  cout << "\nNote: unlike equally spaced (Newton-Cotes) rules, which diverge for\n"
       << "this integrand, the Gauss-Legendre error decreases monotonically.\n";
  return 0;
}
