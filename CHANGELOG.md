# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
