# GaussQuad C++

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20693630-blue)](https://doi.org/10.5281/zenodo.20693630)

**GaussQuad** is a header-only C++ library for computing
**Gaussian quadrature** rules — the nodes `x_k` and weights `w_k` such that

```
∫ w(x) f(x) dx  ≈  Σ_k w_k f(x_k)
```

is exact for polynomials up to degree `2n-1`.

Currently, it supports the classical orthogonal polynomial families:

| Family                | Interval   | Weight `w(x)`            |
|-----------------------|------------|--------------------------|
| Legendre              | `[-1, 1]`  | `1`                      |
| Laguerre              | `[0, ∞)`   | `e^{-x}`                 |
| Generalized Laguerre  | `[0, ∞)`   | `x^α e^{-x}`             |
| Hermite               | `(-∞, ∞)`  | `e^{-x^2}`               |
| Jacobi                | `[-1, 1]`  | `(1-x)^α (1+x)^β`        |

Originally, it was written for my research purposes and used the Newton-Raphson
method to find the nodes, starting from initial seeds for the nodes of the corresponding orthogonal polynomials.
In practice, this works very well for Legendre polynomials (see e.g. M. Newman, *Computational Physics*, 2013)
but not as well for other orthogonal polynomials, where already for small `n` Newton-Raphson can converge to
identical roots for different initial seeds.
For this reason the default solver was eventually changed to the full
[Golub–Welsch algorithm](https://en.wikipedia.org/wiki/Gaussian_quadrature). It first used the Eigen library,
but was later rewritten — with the help of AI coding assistants — in pure C++ using an embedded implicit-shift
tridiagonal **QL** eigensolver that tracks only the first row of the eigenvector matrix needed to compute the
weights.
The code has no external dependencies and is self-contained, and stays accurate to machine precision for high
orders (`n` in the thousands).

A Newton-Raphson solver with asymptotic seeds is also provided for Legendre.

## Prerequisites

A compiler with C++11 support. CMake v3.14+ to build the examples and tests.

## Usage

Include the umbrella header [`gaussquad/gaussquad.h`](include/gaussquad/gaussquad.h)
and call `GaussianQuadrature(n, &poly)` with one of the polynomial families:

```cpp
#include <gaussquad/gaussquad.h>
using namespace gaussquad;

LaguerrePolynomial poly;                 // weight e^{-x} on [0, ∞)
auto rule = GaussianQuadrature(64, &poly);
const std::vector<double>& x = rule.first;   // nodes
const std::vector<double>& w = rule.second;  // weights

// ∫_0^∞ e^{-x} f(x) dx ≈ Σ_k w[k] * f(x[k])
double integral = 0.0;
for (size_t k = 0; k < x.size(); ++k)
    integral += w[k] * f(x[k]);
```

See [`examples/example_quadratures.cpp`](examples/example_quadratures.cpp) for a
full demonstration that prints nodes and weights for every family.

## Building

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build      # run the unit tests
```

### Installing and using from another project

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build
cmake --install build
```

Then, from a downstream CMake project:

```cmake
find_package(GaussQuad REQUIRED)
target_link_libraries(your_app PRIVATE GaussQuad::GaussQuad)
```

### Embedding via add_subdirectory

```cmake
add_subdirectory(gaussquad-cpp)
target_link_libraries(your_app PRIVATE GaussQuad::GaussQuad)
```

When embedded this way, examples, tests, and install rules are off by default
(they default to on only for a standalone/main-project build). Toggle them with
`-DGAUSSQUAD_BUILD_EXAMPLES=ON`, `-DGAUSSQUAD_INCLUDE_TESTS=ON`, `-DGAUSSQUAD_INSTALL=ON`.

## Future plans

Currently only the classical orthogonal polynomial families are implemented.
In principle, the method can be generalized to any family of orthogonal polynomials on a given interval `(a, b)` with a
given weight function `w(x)` and recurrence relation for the same Golub–Welsch solver. Extending the library
in this direction is a viable future plan.

## License

MIT, see [LICENSE](LICENSE).
