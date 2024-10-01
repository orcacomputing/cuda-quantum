
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
// #include "cudaq/qis/qarray.h"
#include "cudaq/qis/qvector.h"
#include <vector>

namespace cudaq {
/// @brief The `plus` gate
// U|0> -> |1>, U|1> -> |2>, ..., and U|d> -> |0>
template <std::size_t Levels>
void plus(qudit<Levels> &q) {
  getExecutionManager()->apply("plus", {}, {}, {{q.n_levels(), q.id()}});
}

/// @brief The `phase shift` gate
template <std::size_t Levels>
void phase_shift(qudit<Levels> &q, const double &angle) {
  getExecutionManager()->apply("phase_shift", {angle}, {},
                               {{q.n_levels(), q.id()}});
}

/// @brief The `beam splitter` gate
template <std::size_t Levels>
void beam_splitter(qudit<Levels> &q, qudit<Levels> &r, const double &angle) {
  getExecutionManager()->apply(
      "beam_splitter", {angle}, {},
      {{q.n_levels(), q.id()}, {r.n_levels(), r.id()}});
}

/// @brief Measure a qudit
template <std::size_t Levels>
int mz(qudit<Levels> &q) {
  return getExecutionManager()->measure({q.n_levels(), q.id()});
}

/// @brief Measure a vector of qudits
template <std::size_t Levels>
std::vector<int> mz(qvector<Levels> &q) {
  std::vector<int> ret;
  for (auto &qq : q)
    ret.emplace_back(mz(qq));
  return ret;
}
} // namespace cudaq
