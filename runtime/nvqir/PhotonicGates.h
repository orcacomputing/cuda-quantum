/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
// #include "nvqir/Gates.h"

#include "cudaq/operators.h"

#include <Eigen/Dense>
#include <vector>

namespace nvqir {

// template <typename Scalar = double>
// static constexpr std::complex<Scalar> im = std::complex<Scalar>(0, 1.);

// template <typename ScalarType = double>
// using ComplexT = std::complex<ScalarType>;

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
extract_flat_matrix(cudaq::complex_matrix &matrix) {
  std::vector<std::complex<Scalar>> flat_matrix;

  auto nrows = matrix.as_eigen().rows();

  cudaq::complex_matrix::value_type *row_repr =
      matrix.get_data(cudaq::complex_matrix::order::row_major);

  for (std::size_t i = 0; i < nrows * nrows; i++) {
    flat_matrix.push_back(static_cast<std::complex<Scalar>>(row_repr[i]));
  }
  return flat_matrix;
};

template <typename Scalar>
cudaq::complex_matrix
displacement_matrix(std::size_t numLevels,
                    std::complex<Scalar> displacement_amplitude) {
  // Returns the displacement operator matrix.
  //  Args:
  //   - numLevels: Number of levels in the qudit.
  //   - displacement: Amplitude of the displacement operator.
  // See also https://en.wikipedia.org/wiki/Displacement_operator.

  auto create = cudaq::boson_op::create(0);
  auto annihilate = cudaq::boson_op::annihilate(0);

  auto term1 = displacement_amplitude * create;
  auto term2 = std::conj(displacement_amplitude) * annihilate;
  auto matrix = (term1 - term2).to_matrix({{0, numLevels}}).exponential();

  return matrix;
};

template <typename Scalar>
cudaq::complex_matrix beam_splitter_matrix(std::size_t numLevels,
                                           Scalar &theta) {
  // Returns the beam splitter operator matrix.
  //  Args:
  //   - numLevels: Number of levels in the qudit.
  //   - theta: Beam splitter angle.
  cudaq::dimension_map dimension_map = {{0, numLevels}, {1, numLevels}};
  static constexpr std::complex<Scalar> im = std::complex<Scalar>(0, 1.);

  auto create0 = cudaq::boson_op::create(0);
  auto annihilate0 = cudaq::boson_op::annihilate(0);
  auto create1 = cudaq::boson_op::create(1);
  auto annihilate1 = cudaq::boson_op::annihilate(1);

  auto term1 = create0 * annihilate1;
  auto term2 = annihilate0 * create1;
  auto matrix =
      (im * theta * (term1 + term2)).to_matrix(dimension_map).exponential();
  return matrix;
};

template <typename Scalar>
cudaq::complex_matrix phase_shift_matrix(std::size_t numLevels, Scalar &phi) {
  // Returns the phase shift operator matrix.
  //  Args:
  //   - numLevels: Number of levels in the qudit.
  //   - phi: Phase shift angle.
  static constexpr std::complex<Scalar> im = std::complex<Scalar>(0, 1.);

  auto create = cudaq::boson_op::create(0);
  auto annihilate = cudaq::boson_op::annihilate(0);

  auto number_op = create * annihilate;
  auto matrix =
      (im * phi * number_op).to_matrix({{0, numLevels}}).exponential();
  return matrix;
};

/// @brief Given the gate name (an element of the GateName enum),
/// return the matrix data, optionally parameterized by a rotation angle.
template <typename Scalar>
std::vector<std::complex<Scalar>>
getPhotonicGateByName(PhotonicGateName name, const std::size_t levels,
                      std::vector<Scalar> angles = {}) {
  switch (name) {

  case (PhotonicGateName::CreateGate): {
    auto length = levels * levels;
    std::vector<std::complex<Scalar>> u(length, 0.0);
    u.at(length - 1) = 1.;
    for (std::size_t i = 1; i < levels; i++) {
      u.at(i * levels + (i - 1)) = 1.;
    }
    return u;
  }
  case (PhotonicGateName::AnnihilateGate): {
    auto length = levels * levels;
    std::vector<std::complex<Scalar>> u(length, 0.0);
    u.at(0) = 1.;
    for (std::size_t i = 0; i < levels - 1; i++) {
      u.at(i * levels + (i + 1)) = 1.;
    }
    return u;
  }
  case (PhotonicGateName::PlusGate): {
    auto length = levels * levels;
    std::vector<std::complex<Scalar>> u(length, 0.0);
    u.at(levels - 1) = 1.;
    for (std::size_t i = 1; i < levels; i++) {
      u.at(i * levels + (i - 1)) = 1.;
    }
    return u;
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
