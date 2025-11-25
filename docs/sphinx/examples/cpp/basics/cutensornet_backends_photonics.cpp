/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile with:
// ```
// nvq++ cutensornet_backends_photonics.cpp -o dyn.x --target tensornet
// ```
//
// This example is meant to demonstrate the `cuTensorNet`
// multi-node/multi-GPU backend.
// On a multi-GPU system, we can enable distributed parallelization across MPI
// processes by initializing MPI (see code) and launch the compiled executable
// with MPI.
// ```
// mpirun -np <N> ./dyn.x
// ```

#include "cudaq.h"
// #include "cudaq/qis/qvector.h"

struct photonicsKernel {
  void operator()() __qpu__ {
    cudaq::qvector<4> qumodes(2);
    // annihilate(qumodes[0]);
    create(qumodes[0]);
    create(qumodes[0]);
    create(qumodes[1]);
    // mz(qumodes);
  }
};

int main() {

  auto state = cudaq::get_state(photonicsKernel{});
  state.dump();
  // auto counts = cudaq::sample(100, photonicsKernel{});
  // counts.dump();

  return 0;
}
