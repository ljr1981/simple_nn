# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-10-08

### Fixed

- XOR did not learn: over 5000 epochs the loss went from 0.256 to 0.2525 and all four predictions stayed at 0.50, while the tests reported PASS. Backprop was not at fault; a finite-difference check matches it to 1e-10. The cause was DENSE_LAYER's initialization. It was documented as Xavier but drew weights uniformly from +/- sqrt(2/fan_in)/2, which is about 3x too small for the deeper layers (8->4: +/-0.25 against Glorot's +/-0.71). Every layer also reused RANDOM's default seed, so they all drew the same sequence. Stacked sigmoids with weights that small sit on a flat plateau. Across 15 seeds at an equal rate, the old scale learned XOR 1 time and Glorot uniform learned it 14 times. `make` now uses Glorot uniform, +/- sqrt(6/(fan_in+fan_out)), with a fresh seed for each layer. The new `make_seeded` gives reproducible weights; it requires seed > 0, because seed 0 keeps RANDOM at 0 and makes every weight equal.
- The MSE gradient now matches the loss it reports: the output gradient is 2(output - target)/n, where it used to be (output - target). The old gradient was half the true one for a single output, and wrong by a further factor of n with several outputs. Equal `compile` rates now take twice the step on single-output networks.
- Mini-batch training used only the last sample of each batch, because `backward` overwrote the gradients instead of adding to them. Gradients now accumulate, and each batch is applied once with rate / batch length (the batch mean). `update_weights` then clears them.
- The `bias_gradients` array was an alias of the caller's gradient array, and `last_input` aliased the caller's input. Both layers now keep their own copies.

### Added

- Gradient queries: `NEURAL_NETWORK.loss` and `compute_gradients`; `DENSE_LAYER.weight`, `bias_value`, `weight_gradient`, `bias_gradient`, `accumulated_samples`, `set_weight`, `set_bias` and `seed`; `LAYER.has_cached_input` and `clear_gradients`; and `NEURAL_NETWORK.input_size`, `output_size`, `is_compiled` and `learning_rate` (now exported).
- Contracts: `backward` requires a prior `forward`. `fit` requires a compiled network with layers. `get_layer` checks its upper bound. DENSE_LAYER's invariant checks the weight, bias and gradient shapes.
- Tests that assert learning. Seeded XOR (2-8-4-1 with SGD, and 2-8-1 with a full batch) must put every prediction within 0.2 of its target and bring the loss below 0.01. A numerical gradient check covers all 26 parameters of a 3-4-2 tanh/sigmoid net (max relative error < 1e-6). A gradient-accumulation test is also new, and the AND and weight-update diagnostics now assert.
- The test target never switched contracts on (no `<assertions>` in the ECF). It now enables precondition, postcondition, check, invariant, loop and supplier_precondition; the suite passes with them on. The XOR tests built `ARRAY [ARRAY [REAL_64]]` with `make`, which violated ARRAY's has_default precondition; they now use `make_filled`, and the test runner counts a violation as a failure instead of crashing.

## [0.1.0] - 2026-01-30

### Added
- Initial release of neural network library
- Core layer abstraction (LAYER deferred class)
- DENSE_LAYER: Fully connected layers with Xavier initialization
- ACTIVATION_LAYER: Sigmoid, ReLU, tanh activation functions
- NEURAL_NETWORK: Network orchestrator with compile/fit/predict API
- TRAINING_RESULT: Loss history tracking
- SIMPLE_NN: Factory class for layer creation
- Full backpropagation with proper gradient computation
- XOR integration test demonstrating non-linear learning
- Comprehensive unit tests

### Features
- Real mathematical implementations (no approximations)
- Design by Contract preconditions/postconditions
- Void-safe code (SCOOP compatible)
- Extensible layer architecture
- Configurable learning rates
- Loss tracking per epoch

### Dependencies
- simple_math: Mathematical functions
- simple_linalg: Linear algebra (ARRAY2)
- base: ISE Standard Library

### Testing
- 1/1 integration tests passing
- XOR problem learning verification
- Framework operational demonstration
