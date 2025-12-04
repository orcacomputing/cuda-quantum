/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/matrix.h"
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <vector>

namespace nvqir {

template <typename ScalarType = double>
using ComplexT = std::complex<ScalarType>;

/// @brief Enumeration of supported CUDA-Q photonic operations
enum class PhotonicGateName {
  CreateGate,
  AnnihilateGate,
  PlusGate,
  BeamSplitterGate,
  PhaseShiftGate,
};

template <typename Scalar>
std::vector<std::complex<Scalar>>
extract_flat_matrix(cudaq::complex_matrix::EigenMatrix &matrix) {
  auto *flat_matrix_ptr = matrix.data();
  std::size_t size = matrix.rows() * matrix.cols();
  std::vector<std::complex<Scalar>> flat_matrix(flat_matrix_ptr,
                                                flat_matrix_ptr + size);
  return flat_matrix;
};

template <typename Scalar>
cudaq::complex_matrix::EigenMatrix annihilate_matrix(std::size_t numLevels) {
  auto matrix = cudaq::complex_matrix(numLevels, numLevels);
  matrix[{0, 0}] = 1.;
  for (std::size_t i = 0; i < numLevels - 1; i++) {
    matrix[{i, i + 1}] = 1.;
  }
  return matrix.as_eigen();
};

template <typename Scalar>
cudaq::complex_matrix::EigenMatrix create_matrix(std::size_t numLevels) {
  auto matrix = cudaq::complex_matrix(numLevels, numLevels);
  matrix[{numLevels - 1, numLevels - 1}] = 1.;
  for (std::size_t i = 1; i < numLevels; i++) {
    matrix[{i, i - 1}] = 1.;
  }
  return matrix.as_eigen();
};

template <typename Scalar>
cudaq::complex_matrix::EigenMatrix plus_matrix(std::size_t numLevels) {
  auto matrix = cudaq::complex_matrix(numLevels, numLevels);
  matrix[{0, numLevels - 1}] = 1.;
  for (std::size_t i = 1; i < numLevels; i++) {
    matrix[{i, i - 1}] = 1.;
  }
  return matrix.as_eigen();
};

template <typename Scalar>
cudaq::complex_matrix::EigenMatrix
beam_splitter_matrix(std::size_t numLevels, double theta, double phi = M_PI_2) {

  using value_type = std::complex<double>;

  std::vector<value_type> sqrt_vals(numLevels);
  for (std::size_t i = 0; i < numLevels; ++i)
    sqrt_vals[i] = std::sqrt(static_cast<value_type>(i));

  value_type ct = std::cos(theta);
  value_type st = std::sin(theta) * std::exp(value_type(0, phi));

  auto R = cudaq::complex_matrix(4, 4);
  R[{0, 2}] = ct;
  R[{0, 3}] = -std::conj(st);
  R[{1, 2}] = st;
  R[{1, 3}] = ct;
  R[{2, 0}] = ct;
  R[{2, 1}] = st;
  R[{3, 0}] = -std::conj(st);
  R[{3, 1}] = ct;

  auto dim = numLevels * numLevels;
  auto matrix = cudaq::complex_matrix(dim, dim);
  matrix[{0, 0}] = 1.0;

  // rank 3
  for (std::size_t m = 0; m < numLevels; ++m) {
    for (std::size_t n = 0; n < numLevels - m; ++n) {
      auto p = m + n;
      if (0 < p && p < numLevels) {
        auto row = m * numLevels + n;
        auto col = p * numLevels + 0;
        auto row_m1 = (m - 1) * numLevels + n;
        auto col_p1 = (p - 1) * numLevels + 0;
        auto row_n1 = m * numLevels + (n - 1);
        if (m > 0 && p > 0)
          matrix[{row, col}] = R[{0, 2}] * sqrt_vals[m] / sqrt_vals[p] *
                               matrix[{row_m1, col_p1}];
        if (n > 0 && p > 0)
          matrix[{row, col}] += R[{1, 2}] * sqrt_vals[n] / sqrt_vals[p] *
                                matrix[{row_n1, col_p1}];
      }
    }
  }

  // rank 4
  for (std::size_t m = 0; m < numLevels; ++m) {
    for (std::size_t n = 0; n < numLevels; ++n) {
      for (std::size_t p = 0; p < numLevels; ++p) {
        auto q = m + n - p;
        if (0 < q && q < numLevels) {
          auto row = m * numLevels + n;
          auto col = p * numLevels + q;
          auto row_m1 = (m - 1) * numLevels + n;
          auto col_q1 = p * numLevels + (q - 1);
          auto row_n1 = m * numLevels + (n - 1);
          if (m > 0 && q > 0)
            matrix[{row, col}] = R[{0, 3}] * sqrt_vals[m] / sqrt_vals[q] *
                                 matrix[{row_m1, col_q1}];
          if (n > 0 && q > 0)
            matrix[{row, col}] += R[{1, 3}] * sqrt_vals[n] / sqrt_vals[q] *
                                  matrix[{row_n1, col_q1}];
        }
      }
    }
  }
  return matrix.as_eigen();
};

template <typename Scalar>
cudaq::complex_matrix::EigenMatrix
displacement_matrix(std::size_t numLevels, double amplitude, double angle) {

  using value_type = std::complex<double>;

  std::vector<value_type> sqrt_vals(numLevels);
  for (int i = 0; i < numLevels; ++i)
    sqrt_vals[i] = std::sqrt(static_cast<double>(i));

  value_type mu0 = amplitude * std::exp(value_type(0, angle));
  value_type mu1 = -amplitude * std::exp(value_type(0, -angle));

  auto matrix = cudaq::complex_matrix(numLevels, numLevels);
  matrix[{0, 0}] = std::exp(-0.5 * amplitude * amplitude);

  for (int m = 1; m < numLevels; ++m)
    matrix[{m, 0}] = mu0 / sqrt_vals[m] * matrix[{m - 1, 0}];

  for (int m = 0; m < numLevels; ++m) {
    for (int n = 1; n < numLevels; ++n) {
      auto term1 = mu1 / sqrt_vals[n] * matrix[{m, n - 1}];
      auto term2 = (m > 0 ? sqrt_vals[m] / sqrt_vals[n] * matrix[{m - 1, n - 1}]
                          : value_type(0.0, 0.0));
      matrix[{m, n}] = term1 + term2;
    }
  }
  return matrix.as_eigen();
};

template <typename Scalar>
cudaq::complex_matrix::EigenMatrix phase_shift_matrix(std::size_t numLevels,
                                                      double phi) {
  static constexpr std::complex<double> im = std::complex<double>(0, 1.);

  auto matrix = cudaq::complex_matrix(numLevels, numLevels);
  for (size_t n = 0; n < numLevels; n++) {
    matrix[{n, n}] = std::exp(n * phi * im);
  }
  return matrix.as_eigen();
};

/// @brief Given the gate name (an element of the GateName enum),
/// return the matrix data, optionally parameterized by a rotation angle.
template <typename Scalar>
std::vector<std::complex<Scalar>>
getPhotonicGateByName(PhotonicGateName name, const std::size_t levels,
                      std::vector<Scalar> angles = {}) {
  switch (name) {

  case (PhotonicGateName::CreateGate): {
    auto create_mat = create_matrix<Scalar>(levels);
    return extract_flat_matrix<Scalar>(create_mat);
  }

  case (PhotonicGateName::AnnihilateGate): {
    auto annihilate_mat = annihilate_matrix<Scalar>(levels);
    return extract_flat_matrix<Scalar>(annihilate_mat);
  }

  case (PhotonicGateName::PlusGate): {
    auto plus_mat = plus_matrix<Scalar>(levels);
    return extract_flat_matrix<Scalar>(plus_mat);
  }

  case (PhotonicGateName::BeamSplitterGate): {
    auto theta = angles[0];
    auto bs_matrix = beam_splitter_matrix<Scalar>(levels, theta);
    return extract_flat_matrix<Scalar>(bs_matrix);
  }

  case (PhotonicGateName::PhaseShiftGate): {
    auto phi = angles[0];
    auto ps_matrix = phase_shift_matrix<Scalar>(levels, phi);
    return extract_flat_matrix<Scalar>(ps_matrix);
  }
  }

  throw std::runtime_error("Invalid gate provided to getGateByName.");
}

/// @brief The create operation as a type. Can instantiate and request
/// its matrix data.
template <typename ScalarType = double>
struct create {
  auto getGate(const std::size_t levels, std::vector<ScalarType> angles = {}) {
    return getPhotonicGateByName<ScalarType>(PhotonicGateName::CreateGate,
                                             levels);
  }
  const std::string name() const { return "create"; }
};

/// @brief The annihilate operation as a type. Can instantiate and request
/// its matrix data.
template <typename ScalarType = double>
struct annihilate {
  auto getGate(const std::size_t levels, std::vector<ScalarType> angles = {}) {
    return getPhotonicGateByName<ScalarType>(PhotonicGateName::AnnihilateGate,
                                             levels);
  }
  const std::string name() const { return "annihilate"; }
};

/// @brief The plus operation as a type. Can instantiate and request
/// its matrix data.
template <typename ScalarType = double>
struct plus {
  auto getGate(const std::size_t levels, std::vector<ScalarType> angles = {}) {
    return getPhotonicGateByName<ScalarType>(PhotonicGateName::PlusGate,
                                             levels);
  }
  const std::string name() const { return "plus"; }
};

/// The Beam Splitter Gate
template <typename ScalarType = double>
struct beam_splitter {
  std::vector<ComplexT<ScalarType>>
  getGate(const std::size_t levels, std::vector<ScalarType> angles = {}) {
    return getPhotonicGateByName<ScalarType>(PhotonicGateName::BeamSplitterGate,
                                             levels, angles);
  }
  const std::string name() const { return "beam_splitter"; }
};

/// The Phase Shift Gate
template <typename ScalarType = double>
struct phase_shift {
  std::vector<ComplexT<ScalarType>>
  getGate(const std::size_t levels, std::vector<ScalarType> angles = {}) {
    return getPhotonicGateByName<ScalarType>(PhotonicGateName::PhaseShiftGate,
                                             levels, angles);
  }
  const std::string name() const { return "phase_shift"; }
};

} // namespace nvqir
