# Simple_NN: Neural Network Library for Eiffel

Production-ready neural network library with real backpropagation for the Eiffel programming language.

## Features

- **Real Backpropagation**: Proper gradient computation via chain rule
- **Layer Abstraction**: Extensible design for custom layer types
- **Multiple Activations**: Sigmoid, ReLU, tanh with derivatives
- **Weight Initialization**: Glorot (Xavier) uniform, a fresh seed for each layer, and `make_seeded` for reproducible runs
- **Verified Gradients**: a numerical gradient check in the test suite
- **Mini-batch Training**: `fit_with_batch_size` averages gradients over each batch
- **Training Utilities**: Loss tracking, configurable learning rates
- **Pure Eiffel**: No external dependencies (uses simple_math, simple_linalg)

## Quick Start

```eiffel
-- Create network
create network.make
network.add_layer (create {DENSE_LAYER}.make_seeded (2, 8, 404))  -- or make (2, 8)
network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (8))
network.add_layer (create {DENSE_LAYER}.make_seeded (8, 1, 505))
network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (1))
network.compile (0.5)  -- learning_rate = 0.5

-- Train on XOR problem
x_train := <<0.0, 0.0>>, <<0.0, 1.0>>, <<1.0, 0.0>>, <<1.0, 1.0>>
y_train := <<0.0>>, <<1.0>>, <<1.0>>, <<0.0>>
result := network.fit (x_train, y_train, 5000)

-- Predict
output := network.predict (<<0.0, 1.0>>)
```

## Architecture

### Core Classes

- **LAYER**: Deferred base class defining forward/backward interface
- **DENSE_LAYER**: Fully connected layer with learnable weights/biases
- **ACTIVATION_LAYER**: Element-wise activation functions
- **NEURAL_NETWORK**: Network orchestrator and training loop
- **TRAINING_RESULT**: Loss history and training metrics
- **SIMPLE_NN**: Factory methods for layer creation

### Design

```
Input → Dense(2→4) → Sigmoid → Dense(4→1) → Sigmoid → Output
```

Each layer supports:
- `forward(input)`: Compute outputs
- `backward(gradient)`: Add this sample's gradients to the accumulators; return the input gradient
- `update_weights(learning_rate)`: Gradient descent step, then clear the gradients

## Dependencies

- **simple_math**: Exponential, sqrt, log, trigonometric functions
- **simple_linalg**: Matrix operations (ARRAY2)
- **base**: ISE Standard Library

## Building

```bash
cd simple_nn
/d/prod/ec.sh test -config simple_nn.ecf -target simple_nn_tests
```

## Testing

```bash
./EIFGENs/simple_nn_tests/F_code/simple_nn.exe
```

**Test Results** (0.1.1, full contracts on): 7/7 passed
- XOR, seeded 2-8-4-1 with SGD: every prediction within 0.2 of its target, final loss < 0.01 (0.0003)
- XOR, seeded 2-8-1 with a full batch: the same criteria (0.0006)
- Numerical gradient check: all 26 parameters of a 3-4-2 tanh/sigmoid net match central differences (max relative error 3e-9)
- Batch gradient accumulation, AND gate, weight updates, loss computation

## API Reference

### NEURAL_NETWORK

```eiffel
-- Configuration
add_layer (layer: LAYER)
compile (learning_rate: REAL_64)

-- Training
fit (x_train, y_train: ARRAY; epochs: INTEGER): TRAINING_RESULT
fit_with_batch_size (x_train, y_train: ARRAY; epochs, batch_size: INTEGER): TRAINING_RESULT

-- Gradients
loss (input, target: ARRAY): REAL_64            -- MSE for one sample
compute_gradients (input, target: ARRAY)        -- fill each dense layer's gradients

-- Prediction
predict (input: ARRAY): ARRAY

-- Queries
input_size, output_size, layer_count: INTEGER
learning_rate: REAL_64
is_compiled: BOOLEAN
get_layer (index: INTEGER): LAYER
```

### DENSE_LAYER

```eiffel
make (input_size, output_size: INTEGER)              -- Glorot uniform, a fresh seed
make_seeded (input_size, output_size, seed: INTEGER) -- reproducible weights (seed > 0)
weight (row, column), bias_value (row): REAL_64
weight_gradient (row, column), bias_gradient (row): REAL_64
accumulated_samples: INTEGER
set_weight (row, column, value), set_bias (row, value)
forward (input: ARRAY): ARRAY
backward (gradient: ARRAY): ARRAY
update_weights (learning_rate: REAL_64)
clear_gradients
```

### ACTIVATION_LAYER

```eiffel
make_sigmoid (size: INTEGER)
make_relu (size: INTEGER)
make_tanh (size: INTEGER)
```

## Implementation Details

### Backpropagation

Forward pass computes: `output = activation(weights @ input + bias)`

Loss is mean squared error, `L = (1/n) Σ (output - target)²`, so backprop starts from `∂L/∂output = 2 (output - target) / n`.

Backward pass computes:
1. Input gradient: `∂L/∂input = weights^T @ ∂L/∂output`
2. Weight gradient: `∂L/∂weights = ∂L/∂output @ input^T`
3. Bias gradient: `∂L/∂bias = ∂L/∂output`

### Weight Updates

Gradient descent: `w := w - learning_rate * gradient`. `fit` updates after every sample (SGD). `fit_with_batch_size` sums each batch's gradients and applies them once, with rate `learning_rate / batch_length` (the batch mean).

### Initialization

Glorot (Xavier) uniform: `w ~ U(-limit, limit)` with `limit = sqrt (6 / (fan_in + fan_out))`, biases 0. Each layer built with `make` takes its own seed. Use `make_seeded` for reproducible runs.

### Activation Functions

- **Sigmoid**: σ(z) = 1/(1 + exp(-z))
- **ReLU**: max(0, z)
- **Tanh**: (exp(z) - exp(-z)) / (exp(z) + exp(-z))

## Performance Considerations

- **Gradient Computation**: O(layer_size²) per layer
- **Training**: O(epochs * samples * network_size²)
- **Memory**: O(weights + activations) for full network

## Future Enhancements

- [x] Batch processing (mini-batch gradient descent)
- [ ] Convolutional layers
- [ ] Recurrent layers (LSTM, GRU)
- [ ] Batch normalization
- [ ] Layer normalization
- [ ] Dropout regularization
- [ ] Different optimizers (Adam, RMSprop, Momentum)
- [ ] Model serialization/deserialization
- [ ] GPU acceleration

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! Please follow Eiffel coding standards and include tests.

## See Also

- [Simple_ML](https://github.com/simple-eiffel/simple_ml) - Machine learning algorithms
- [Simple_Math](https://github.com/simple-eiffel/simple_math) - Mathematical functions
- [Simple_Linalg](https://github.com/simple-eiffel/simple_linalg) - Linear algebra
