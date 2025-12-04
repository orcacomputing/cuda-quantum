/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SampleResult.h"
#include "cudaq/host_config.h"
#include "cudaq/operators.h"
#include "cudaq/platform.h"
#include "cudaq/qis/modifiers.h"
#include "cudaq/qis/pauli_word.h"
#include "cudaq/qis/qarray.h"
#include "cudaq/qis/qkernel.h"
#include "cudaq/qis/qreg.h"
#include "cudaq/qis/qudit.h"
#include "cudaq/qis/qvector.h"
#include <algorithm>
#include <cstring>
#include <functional>

// This file describes the API for a default qudit logical instruction
// set for CUDA-Q kernels.

namespace cudaq {

// Define the common single qudit operations.
namespace qudit_op {
#define ConcreteQuditOp(NAME)                                                  \
  struct NAME##Op {                                                            \
    static const std::string name() { return #NAME; }                          \
  };

ConcreteQuditOp(create) ConcreteQuditOp(annihilate) ConcreteQuditOp(phase_shift)
    ConcreteQuditOp(beam_splitter)

} // namespace qudit_op

// Convert a qudit to its unique id representation
template <std::size_t Levels>
inline QuditInfo quditToQuditInfo(qudit<Levels> &q) {
  return {q.n_levels(), q.id()};
}

// TODO can qudit is_negative be generalized?
template <std::size_t Levels>
inline bool quditIsNegative(qudit<Levels> &q) {
  return q.is_negative();
}

#if CUDAQ_USE_STD20

/// C++20 variant using templates.

/// This function will apply the specified `QuantumOp`. It will check the
/// modifier template type and if it is `base`, it will apply the operation to
/// any qudits provided as input.
template <typename QuantumOp, typename mod = base, typename... QuditArgs>
void oneQuditApply(QuditArgs &...args) {
  // Get the name of this operation
  auto gateName = QuantumOp::name();

  // Get the number of input qudits
  constexpr std::size_t nArgs = sizeof...(QuditArgs);

  // Map the qudits to their unique ids and pack them.
  std::vector<QuditInfo> quditInfos{quditToQuditInfo(args)...};
  std::vector<bool> quditIsNegated{quditIsNegative(args)...};

  // If mod == base, then we just want to apply the gate to all qudits provided.
  // This is a broadcast application.
  if constexpr (std::is_same_v<mod, base>) {
    for (auto &qudit : quditInfos)
      getExecutionManager()->apply(gateName, {}, {}, {qudit});

    // Nothing left to do, return
    return;
  }

  // TODO: do we need control negation handling for qudits?
}

#define CUDAQ_QIS_PHOTONIC_ONE_TARGET_QUDIT_(NAME)                             \
  namespace types {                                                            \
  struct NAME {                                                                \
    inline static const std::string name{#NAME};                               \
  };                                                                           \
  }                                                                            \
  template <typename mod = base, typename... QuditArgs>                        \
  void NAME(QuditArgs &...args) {                                              \
    oneQuditApply<qudit_op::NAME##Op, mod>(args...);                           \
  }                                                                            \
  template <typename mod = base, typename QuditRange>                          \
    requires(std::ranges::range<QuditRange>)                                   \
  void NAME(QuditRange &qr) {                                                  \
    for (auto &q : qr) {                                                       \
      NAME<mod>(q);                                                            \
    }                                                                          \
  }                                                                            \
  template <typename mod = base, typename QuditRange>                          \
    requires(std::ranges::range<QuditRange>)                                   \
  void NAME(QuditRange &&qr) {                                                 \
    for (auto &q : qr) {                                                       \
      NAME<mod>(q);                                                            \
    }                                                                          \
  }

#else // not C++20

/// C++17 variant does NOT use templates.
template <typename QuantumOp, typename... QuditArgs>
void oneQuditApply(QuditArgs &...args) {
  // Get the name of this operation
  auto gateName = QuantumOp::name();

  // Map the qudits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> quditInfos{quditToQuditInfo(args)...};

  // If there are more than one qudits, then we just want to apply the gate to
  // all qudits provided
  for (auto &qudit : quditInfos)
    getExecutionManager()->apply(gateName, {}, {}, {qudit});
}

#define CUDAQ_QIS_PHOTONIC_ONE_TARGET_QUDIT_(NAME)                             \
  namespace types {                                                            \
  struct NAME {                                                                \
    inline static const std::string name{#NAME};                               \
  };                                                                           \
  }                                                                            \
  template <typename... QuditArgs>                                             \
  void NAME(QuditArgs &...args) {                                              \
    oneQuditApply<qudit_op::NAME##Op>(args...);                                \
  }                                                                            \
  template <typename QuditRange, std::size_t Levels,                           \
            typename = std::enable_if_t<!std::is_same_v<                       \
                std::remove_reference_t<std::remove_cv_t<QuditRange>>,         \
                cudaq::qudit<Levels>>>>                                        \
  void NAME(QuditRange &qr) {                                                  \
    for (auto &q : qr) {                                                       \
      NAME(q);                                                                 \
    }                                                                          \
  }                                                                            \
  template <typename QuditRange, std::size_t Levels,                           \
            typename = std::enable_if_t<!std::is_same_v<                       \
                std::remove_reference_t<std::remove_cv_t<QuditRange>>,         \
                cudaq::qudit<Levels>>>>                                        \
  void NAME(QuditRange &&qr) {                                                 \
    for (auto &q : qr) {                                                       \
      NAME(q);                                                                 \
    }                                                                          \
  }

#endif // not C++20

// Instantiate the above functions for the single qudit gates
CUDAQ_QIS_PHOTONIC_ONE_TARGET_QUDIT_(create)
CUDAQ_QIS_PHOTONIC_ONE_TARGET_QUDIT_(annihilate)

#if CUDAQ_USE_STD20

template <typename QuantumOp, typename mod = base, typename ScalarAngle,
          typename... QuditArgs>
void oneQuditSingleParameterApply(ScalarAngle angle, QuditArgs &...args) {
  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the qudits to their unique ids and pack them into a std::array
  constexpr std::size_t nArgs = sizeof...(QuditArgs);
  std::vector<QuditInfo> targets{quditToQuditInfo(args)...};

  // If there are more than one qudits and mod == base, then
  // we just want to apply the same gate to all qudits provided
  if constexpr (nArgs > 1 && std::is_same_v<mod, base>) {
    for (auto &targetId : targets)
      getExecutionManager()->apply(gateName, {angle}, {}, {targetId});

    // Nothing left to do, return
    return;
  }
  // If we are here, then mod must be control or adjoint
  // Extract the controls and the target
  std::vector<QuditInfo> controls(targets.begin(), targets.begin() + nArgs - 1);

  // Apply the gate
  getExecutionManager()->apply(gateName, {angle}, controls, {targets.back()},
                               std::is_same_v<mod, adj>);
}

#define CUDAQ_QIS_PHOTONIC_PARAM_ONE_TARGET_(NAME)                             \
  namespace types {                                                            \
  struct NAME {                                                                \
    inline static const std::string name{#NAME};                               \
  };                                                                           \
  }                                                                            \
  template <typename mod = base, typename ScalarAngle, typename... QuditArgs>  \
  void NAME(ScalarAngle angle, QuditArgs &...args) {                           \
    oneQuditSingleParameterApply<qudit_op::NAME##Op, mod>(angle, args...);     \
  }

#else // not C++20

template <typename QuantumOp, typename ScalarAngle, typename... QuditArgs>
void oneQuditSingleParameterApply(ScalarAngle angle, QuditArgs &...args) {

  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> targets{quditToQuditInfo(args)...};

  // We just want to apply the same gate to all qubits provided
  for (auto &targetId : targets)
    getExecutionManager()->apply(gateName, {angle}, {}, {targetId});
}

#define CUDAQ_QIS_PHOTONIC_PARAM_ONE_TARGET_(NAME)                             \
  namespace types {                                                            \
  struct NAME {                                                                \
    inline static const std::string name{#NAME};                               \
  };                                                                           \
  }                                                                            \
  template <typename ScalarAngle, typename... QuditArgs>                       \
  void NAME(ScalarAngle angle, QuditArgs &...args) {                           \
    oneQuditSingleParameterApply<qudit_op::NAME##Op>(angle, args...);          \
  }

#endif // not C++20

// FIXME add One Qudit Single Parameter Broadcast over register with an angle
// for each qudit

CUDAQ_QIS_PHOTONIC_PARAM_ONE_TARGET_(phase_shift)

#if CUDAQ_USE_STD20

template <typename QuantumOp, typename mod = base, typename ScalarAngle,
          typename... QuditArgs>
void twoQuditSingleParameterApply(ScalarAngle angle, QuditArgs &...args) {
  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the qubits to their unique ids and pack them into a std::array
  constexpr std::size_t nArgs = sizeof...(QuditArgs);
  std::vector<QuditInfo> quditIds{quditToQuditInfo(args)...};

  // If there are more than one qubits and mod == base, then
  // we just want to apply the same gate to all qubits provided
  if constexpr (nArgs == 2) {
    getExecutionManager()->apply(gateName, {angle}, {}, {quditIds});
    // Nothing left to do, return
    return;
  } else {
    static_assert(
        std::is_same_v<mod, ctrl>,
        "More than 2 qudits passed to beam_splitter but modifier != ctrl.");
  }
  // Controls are all qubits except the last 2
  std::vector<QuditInfo> controls(quditIds.begin(),
                                  quditIds.begin() + quditIds.size() - 2);
  std::vector<QuditInfo> targets(quditIds.end() - 2, quditIds.end());

  // Apply the gate
  getExecutionManager()->apply(gateName, {angle}, controls, targets);
}

#define CUDAQ_QIS_PHOTONIC_PARAM_TWO_TARGET_(NAME)                             \
  namespace types {                                                            \
  struct NAME {                                                                \
    inline static const std::string name{#NAME};                               \
  };                                                                           \
  }                                                                            \
  template <typename mod = base, typename ScalarAngle, typename... QuditArgs>  \
  void NAME(ScalarAngle angle, QuditArgs &...args) {                           \
    twoQuditSingleParameterApply<qudit_op::NAME##Op, mod>(angle, args...);     \
  }

#else // not C++20

template <typename QuantumOp, typename ScalarAngle, typename... QuditArgs>
void twoQuditSingleParameterApply(ScalarAngle angle, QuditArgs &...args) {

  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> quditIds{quditToQuditInfo(args)...};

  // We just want to apply the same gate to all qubits provided

  getExecutionManager()->apply(gateName, {angle}, {}, {quditIds});

#define CUDAQ_QIS_PHOTONIC_PARAM_TWO_TARGET_(NAME)                             \
  namespace types {                                                            \
  struct NAME {                                                                \
    inline static const std::string name{#NAME};                               \
  };                                                                           \
  }                                                                            \
  template <typename ScalarAngle, typename... QuditArgs>                       \
  void NAME(ScalarAngle angle, QuditArgs &...args) {                           \
    twoQuditSingleParameterApply<qudit_op::NAME##Op>(angle, args...);          \
  }

#endif // not C++20

CUDAQ_QIS_PHOTONIC_PARAM_TWO_TARGET_(beam_splitter)

/// @brief Measure a qudit
template <std::size_t Levels>
int mpnr(cudaq::qudit<Levels> &q) {
  return cudaq::getExecutionManager()->measure({q.n_levels(), q.id()});
}

/// @brief Measure a vector of qudits
template <std::size_t Levels>
std::vector<int> mpnr(cudaq::qvector<Levels> &q) {
  std::vector<int> ret;
  for (auto &qq : q)
    ret.emplace_back(mpnr(qq));
  return ret;
}

} // namespace cudaq
