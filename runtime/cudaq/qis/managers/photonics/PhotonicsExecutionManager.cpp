/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/Logger.h"
#include "cudaq/operators.h"
#include "cudaq/qis/managers/BasicExecutionManager.h"
#include "cudaq/utils/cudaq_utils.h"
#include "nvqir/PhotonicGates.h"
#include "qpp.h"
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>

namespace cudaq {

struct PhotonicsState : public cudaq::SimulationState {
  /// @brief The state. This class takes ownership move semantics.
  qpp::ket state;

  /// @brief The qudit-levels (`qumodes`)
  std::size_t levels;

  PhotonicsState(qpp::ket &&data, std::size_t lvl)
      : state(std::move(data)), levels(lvl) {}

  /// TODO: Rename the API to be generic
  std::size_t getNumQubits() const override {
    return (std::log2(state.size()) / std::log2(levels));
  }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    throw "not supported for this photonics simulator";
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(fmt::format(
          "[photonics] getAmplitude with an invalid number of bits in the "
          "basis state: expected {}, provided {}.",
          getNumQubits(), basisState.size()));

    // Convert the basis state to an index value
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [&](std::size_t acc, int qudit) { return (acc * levels) + qudit; });
    return state[idx];
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[photonics] invalid tensor requested.");
    return Tensor{
        reinterpret_cast<void *>(
            const_cast<std::complex<double> *>(state.data())),
        std::vector<std::size_t>{static_cast<std::size_t>(state.size())},
        getPrecision()};
  }

  /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0)
      throw std::runtime_error("[photonics] invalid tensor requested.");
    if (indices.size() != 1)
      throw std::runtime_error("[photonics] invalid element extraction.");
    return state[indices[0]];
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    throw "not supported for this photonics simulator";
  }

  void dump(std::ostream &os) const override { os << state << "\n"; }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {
    qpp::ket k;
    state = k;
  }
};

/// @brief The `PhotonicsExecutionManager` implements allocation, deallocation,
/// and quantum instruction application for the photonics execution manager.
class PhotonicsExecutionManager : public cudaq::BasicExecutionManager {
private:
  /// @brief Current state
  qpp::ket state;

  /// @brief The qudit-levels (`qumodes`)
  std::size_t levels;

  /// @brief Instructions are stored in a map
  std::unordered_map<std::string, std::function<void(const Instruction &)>>
      instructions;

  /// @brief Qudits to be sampled
  std::vector<cudaq::QuditInfo> sampleQudits;

  /// @brief Convert internal qudit index to Q++ qudit index.
  ///
  /// In Q++, qudits are indexed from left to right, and thus q0 is the leftmost
  /// qudit. Internally, in CUDA-Q, qudits are index from right to left,
  /// hence q0 is the rightmost qudit. Example:
  /// ```
  ///   Q++ indices:  0  1  2  3
  ///                |0>|0>|0>|0>
  ///                 3  2  1  0 : CUDA-Q indices
  /// ```
  std::size_t convertQuditIndex(std::size_t quditIndex) {
    assert(state.size() > 0 && "The state is empty, and thus has no qudits");
    return std::log2(state.size()) / std::log2(levels) - quditIndex - 1;
  }

protected:
  /// @brief Qudit allocation method: a zeroState is first initialized, the
  /// following ones are added via kron operators
  void allocateQudit(const cudaq::QuditInfo &q) override {
    if (state.size() == 0) {
      // qubit will give [1,0], qutrit will give [1,0,0] and so on...
      state = qpp::ket::Zero(q.levels);
      state(0) = 1.0;
      levels = q.levels;
      return;
    }

    qpp::ket zeroState = qpp::ket::Zero(q.levels);
    zeroState(0) = 1.0;
    state = qpp::kron(state, zeroState);
  }

  /// @brief Allocate a set of `qudits` (`qumodes`) with a single call.
  void allocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    for (auto &q : qudits)
      allocateQudit(q);
  }

  void initializeState(const std::vector<cudaq::QuditInfo> &targets,
                       const void *state,
                       simulation_precision precision) override {
    throw std::runtime_error("initializeState not implemented.");
  }

  virtual void initializeState(const std::vector<QuditInfo> &targets,
                               const SimulationState *state) override {
    throw std::runtime_error("initializeState not implemented.");
  }

  /// @brief Qudit deallocation method
  void deallocateQudit(const cudaq::QuditInfo &q) override {}

  /// @brief Deallocate a set of `qudits` (`qumodes`) with a single call.
  void deallocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {}

  /// @brief Handler for when the photonics execution context changes
  void handleExecutionContextChanged() override {
    if (!executionContext)
      throw std::runtime_error(
          "Execution context is not set for the PhotonicsExecutionManager.");

    if (!(executionContext->name == "sample" ||
          executionContext->name == "extract-state" ||
          executionContext->name == "tracer"))
      throw std::runtime_error(executionContext->name +
                               " is not supported on this target");
  }

  /// @brief Handler for when the current execution context has ended. It
  /// returns samples to the execution context if it is "sample".
  void handleExecutionContextEnded() override {
    if (executionContext) {
      std::vector<std::size_t> ids;
      for (auto &s : sampleQudits) {
        ids.push_back(convertQuditIndex(s.id));
      }
      if (executionContext->name == "sample") {
        CUDAQ_INFO("Sampling");
        auto shots = executionContext->shots;
        auto sampleResult =
            qpp::sample(shots, state, ids, sampleQudits.begin()->levels);
        cudaq::ExecutionResult counts;
        for (auto [result, count] : sampleResult) {
          std::stringstream bitstring;
          for (const auto &quditRes : result) {
            bitstring << quditRes;
          }
          // Add to the sample result
          // in mid-circ sampling mode this will append 1 bitstring
          counts.appendResult(bitstring.str(), count);
          // Reset the string.
          bitstring.str("");
          bitstring.clear();
        }
        executionContext->result.append(counts);
      } else if (executionContext->name == "extract-state") {
        CUDAQ_INFO("Extracting state");
        // If here, then we care about the result qudit, so compute it.
        for (auto &q : sampleQudits) {
          const auto measurement_tuple = qpp::measure(
              state, qpp::cmat::Identity(q.levels, q.levels),
              {convertQuditIndex(q.id)},
              /*qudit dimension=*/q.levels, /*destructive measmt=*/false);
          const auto measurement_result = std::get<qpp::RES>(measurement_tuple);
          const auto &post_meas_states = std::get<qpp::ST>(measurement_tuple);
          const auto &collapsed_state = post_meas_states[measurement_result];
          state = Eigen::Map<const qpp::ket>(collapsed_state.data(),
                                             collapsed_state.size());
        }

        executionContext->simulationState =
            std::make_unique<cudaq::PhotonicsState>(std::move(state), levels);
      }
      // Reset the state and qudits
      state.resize(0);
      sampleQudits.clear();
    }
  }

  /// @brief Method for executing instructions.
  void executeInstruction(const Instruction &instruction) override {
    auto operation = instructions[std::get<0>(instruction)];
    operation(instruction);
  }

  /// @brief Method for performing qudit measurement.
  int measureQudit(const cudaq::QuditInfo &q,
                   const std::string &registerName) override {
    if (executionContext && executionContext->name == "sample") {
      sampleQudits.push_back(q);
      return 0;
    }

    if (executionContext && executionContext->name == "extract-state") {
      sampleQudits.push_back(q);
      return 0;
    }

    // If here, then we care about the result qudit, so compute it.
    const auto measurement_tuple = qpp::measure(
        state, qpp::cmat::Identity(q.levels, q.levels),
        {convertQuditIndex(q.id)},
        /*qudit dimension=*/q.levels, /*destructive measmt=*/false);
    const auto measurement_result = std::get<qpp::RES>(measurement_tuple);
    const auto &post_meas_states = std::get<qpp::ST>(measurement_tuple);
    const auto &collapsed_state = post_meas_states[measurement_result];
    state = Eigen::Map<const qpp::ket>(collapsed_state.data(),
                                       collapsed_state.size());

    CUDAQ_INFO("Measured qubit {} -> {}", convertQuditIndex(q.id),
               measurement_result);
    return measurement_result;
  }

  /// @brief Measure the state in the basis described by the given `spin_op`.
  void measureSpinOp(const cudaq::spin_op &) override {}

  /// @brief Method for performing qudit reset.
  void resetQudit(const cudaq::QuditInfo &id) override {}

public:
  PhotonicsExecutionManager() {

    instructions.emplace("annihilate", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target = qudits[0];
      size_t d = target.levels;
      auto u = nvqir::annihilate_matrix<double>(d);
      CUDAQ_INFO("Applying annihilate on {}<{}>", convertQuditIndex(target.id),
                 target.levels);
      state = qpp::apply(state, u, {convertQuditIndex(target.id)}, d);
    });

    instructions.emplace("create", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target = qudits[0];
      size_t d = target.levels;
      auto u = nvqir::create_matrix<double>(d);
      CUDAQ_INFO("Applying create on {}<{}>", convertQuditIndex(target.id),
                 target.levels);
      state = qpp::apply(state, u, {convertQuditIndex(target.id)}, d);
    });

    instructions.emplace("plus", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target = qudits[0];
      size_t d = target.levels;
      auto u = nvqir::plus_matrix<double>(d);
      CUDAQ_INFO("Applying plus on {}<{}>", convertQuditIndex(target.id),
                 target.levels);
      state = qpp::apply(state, u, {convertQuditIndex(target.id)}, d);
    });

    instructions.emplace("beam_splitter", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target1 = qudits[0];
      auto target2 = qudits[1];
      size_t d = target1.levels;
      double theta = params[0];
      auto BS = nvqir::beam_splitter_matrix<double>(d, theta);
      CUDAQ_INFO("Applying beam_splitter on {}<{}> and {}<{}>",
                 convertQuditIndex(target1.id), target1.levels,
                 convertQuditIndex(target2.id), target2.levels);
      state = qpp::apply(
          state, BS,
          {convertQuditIndex(target1.id), convertQuditIndex(target2.id)}, d);
    });

    instructions.emplace("phase_shift", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target = qudits[0];
      size_t d = target.levels;
      double phi = params[0];
      qpp::cmat PS = nvqir::phase_shift_matrix<double>(d, phi);
      CUDAQ_INFO("Applying phase_shift on {}<{}>", convertQuditIndex(target.id),
                 target.levels);
      state =
          qpp::apply(state, PS, {convertQuditIndex(target.id)}, target.levels);
    });
  }

  virtual ~PhotonicsExecutionManager() = default;

  cudaq::SpinMeasureResult measure(const cudaq::spin_op &op) override {
    throw "spin_op observation (cudaq::observe()) is not supported for this "
          "photonics simulator";
  }

}; // PhotonicsExecutionManager

} // namespace cudaq

CUDAQ_REGISTER_EXECUTION_MANAGER(PhotonicsExecutionManager, photonics)
