/**
 * @file ComputeQuadratures.cpp
 * @brief Demonstration program for Gaussian quadrature computation.
 *
 * This program demonstrates the computation of nodes and weights for various
 * types of Gaussian quadrature rules. It shows examples for:
 * - Gauss-Legendre quadrature
 * - Gauss-Laguerre quadrature
 * - Generalized Gauss-Laguerre quadrature
 * - Gauss-Hermite quadrature
 * - Gauss-Jacobi quadrature
 *
 * For each quadrature type, the program prints a table of nodes and weights,
 * including both the original weights and the weights divided by the weight
 * function (showing the "tilde" weights for integration without explicit
 * weight function).
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cassert>
#include "gaussquad/gaussquad.h"

using namespace std;

/**
 * @brief Main function demonstrating Gaussian quadrature computation.
 * 
 * Computes and displays quadrature nodes and weights for different polynomial
 * families using n=16 nodes. Shows both the standard weights and the weights
 * normalized by the weight function.
 * 
 * @return 0 on successful completion
 */
int main() {
  int n = 16;

  cout << "Gauss-Legendre Quadrature" << endl;

  auto Gaussxw = gaussquad::GaussianQuadraturesLegendre(n);

  const int tabsize = 20;
  cout << setw(tabsize) << "k" << " " << setw(tabsize) << "xk" << " " << setw(tabsize) << "wk" << endl;
  cout << setprecision(15);
  double wsum = 0.;
  for(int k = 1; k <= n; ++k) {
    cout << setw(tabsize) << k << " " << setw(tabsize) << Gaussxw.first[k-1] << " " << setw(tabsize) << Gaussxw.second[k-1] << endl;
    wsum += Gaussxw.second[k-1];
  }
  cout << "Sum of weights: " << wsum << endl;

  cout << endl;


  cout << endl;
  cout << "Gauss-Laguerre Quadrature" << endl;

  gaussquad::OrthogonalPolynomial *polyLag = new gaussquad::LaguerrePolynomial();
  vector<double> x0Lag = gaussquad::NodesGolubWelsch(n, polyLag);
  Gaussxw = gaussquad::GaussianQuadraturesNewton(n, x0Lag, polyLag);
  cout << setw(tabsize) << "k" << " "
       << setw(tabsize) << "xk" << " "
       << setw(tabsize) << "wk" << " "
       << setw(tabsize) << "wktil" << endl;
  cout << setprecision(15);
  wsum = 0.;
  for(int k = 1; k <= n; ++k) {
    cout << setw(tabsize) << k << " "
         << setw(tabsize) << Gaussxw.first[k-1] << " "
         << setw(tabsize) << Gaussxw.second[k-1] << " "
         << setw(tabsize) << Gaussxw.second[k-1] / polyLag->Weight(Gaussxw.first[k-1]) << endl;
    wsum += Gaussxw.second[k-1];
  }
  delete polyLag;

  cout << "Sum of weights: " << wsum << endl;

  cout << endl;
  cout << "Gauss-GenLaguerre Quadrature" << endl;
  double Lalpha = 0.5;
  gaussquad::OrthogonalPolynomial *polyLagAl = new gaussquad::GeneralizedLaguerrePolynomial(Lalpha);
  vector<double> x0LagAl = gaussquad::NodesGolubWelsch(n, polyLagAl);
  Gaussxw = gaussquad::GaussianQuadraturesNewton(n, x0LagAl, polyLagAl);
  cout << setw(tabsize) << "k" << " "
       << setw(tabsize) << "xk" << " "
       << setw(tabsize) << "wk" << " "
       << setw(tabsize) << "wktil" << endl;
  cout << setprecision(15);
  wsum = 0.;
  for(int k = 1; k <= n; ++k) {
    cout << setw(tabsize) << k << " "
         << setw(tabsize) << Gaussxw.first[k-1] << " "
         << setw(tabsize) << Gaussxw.second[k-1] << " "
         << setw(tabsize) << Gaussxw.second[k-1] / polyLagAl->Weight(Gaussxw.first[k-1]) << endl;
    wsum += Gaussxw.second[k-1];
  }
  delete polyLagAl;

  cout << "Sum of weights: " << wsum << endl;

  cout << endl;
  cout << "Gauss-Hermite Quadrature" << endl;

  gaussquad::OrthogonalPolynomial *polyHer = new gaussquad::HermitePolynomial();
  vector<double> x0Her = gaussquad::NodesGolubWelsch(n, polyHer);
  Gaussxw = gaussquad::GaussianQuadraturesNewton(n, x0Her, polyHer);
  cout << setw(tabsize) << "k" << " "
       << setw(tabsize) << "xk" << " "
       << setw(tabsize) << "wk" << " "
       << setw(tabsize) << "wktil" << endl;
  cout << setprecision(15);
  wsum = 0.;
  for(int k = 1; k <= n; ++k) {
    cout << setw(tabsize) << k << " "
         << setw(tabsize) << Gaussxw.first[k-1] << " "
         << setw(tabsize) << Gaussxw.second[k-1] << " "
         << setw(tabsize) << Gaussxw.second[k-1] / polyHer->Weight(Gaussxw.first[k-1]) << endl;
    wsum += Gaussxw.second[k-1];
  }
  delete polyHer;

  cout << "Sum of weights: " << wsum << endl;

  cout << endl;
  cout << "Gauss-Jacobi Quadrature" << endl;
  double Jacalpha = 0.5, Jacbeta = 0.75;
//  Jacalpha = Jacbeta = 0.5;
  gaussquad::OrthogonalPolynomial *polyJacobi = new gaussquad::JacobiPolynomial(Jacalpha, Jacbeta);
  vector<double> x0LagJacobi = gaussquad::NodesGolubWelsch(n, polyJacobi);
//  cout << x0LagJacobi[2] << endl;
  Gaussxw = gaussquad::GaussianQuadraturesNewton(n, x0LagJacobi, polyJacobi);
  cout << setw(tabsize) << "k" << " "
       << setw(tabsize) << "xk" << " "
       << setw(tabsize) << "wk" << " "
       << setw(tabsize) << "wktil" << endl;
  cout << setprecision(15);
  wsum = 0.;
  for(int k = 1; k <= n; ++k) {
    cout << setw(tabsize) << k << " "
         << setw(tabsize) << Gaussxw.first[k-1] << " "
         << setw(tabsize) << Gaussxw.second[k-1] << " "
         << setw(tabsize) << Gaussxw.second[k-1] / polyJacobi->Weight(Gaussxw.first[k-1]) << endl;
    wsum += Gaussxw.second[k-1];
  }
  delete polyJacobi;

  cout << "Sum of weights: " << wsum << endl;

  return 0;
}