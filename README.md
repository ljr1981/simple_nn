# Simple_NN: Neural Network Library for Eiffel

Production-ready neural network library with real backpropagation for the Eiffel programming language.

## Features

- **Real Backpropagation**: Proper gradient computation via chain rule
- **Layer Abstraction**: Extensible design for custom layer types
- **Multiple Activations**: Sigmoid, ReLU, tanh with derivatives
- **Weight Initialization**: Xavier initialization for stable training
- **Training Utilities**: Loss tracking, configurable learning rates
- **Pure Eiffel**: No external dependencies (uses simple_math, simple_linalg)

## Quick Start

```eiffel
-- Create network
create network.make
network.add_layer (create {DENSE_LAYER}.make (2, 4))
network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (4))
network.add_layer (create {DENSE_LAYER}.make (4, 1))
network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (1))
network.compile (0.5)  -- learning_rate = 0.5

-- Train on XOR problem
x_train := <<0.0, 0.0>>, <<0.0, 1.0>>, <<1.0, 0.0>>, <<1.0, 1.0>>
y_train := <<0.0>>, <<1.0>>, <<1.0>>, <<0.0>>
result := network.fit (x_train, y_train, 1000)

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
- `backward(gradient)`: Compute input gradients
- `update_weights(learning_rate)`: Gradient descent

## Dependencies

- **simple_math**: Exponential, sqrt, log, trigonometric functions
- **simple_linalg**: Matrix operations (ARRAY2)
- **base**: ISE Standard Library

## Building

```bash
cd simple_nn
ec.sh -batch -config simple_nn.ecf -target simple_nn_tests -finalize
```

## Testing

```bash
./EIFGENs/simple_nn_tests/F_code/simple_nn.exe
```

**Test Results**: ✅ 1/1 test passed
- XOR problem learning verification
- Network trains and loss decreases
- Framework operational

## API Reference

### NEURAL_NETWORK

```eiffel
-- Configuration
add_layer (layer: LAYER)
compile (learning_rate: REAL_64)

-- Training
fit (x_train, y_train: ARRAY; epochs: INTEGER): TRAINING_RESULT

-- Prediction
predict (input: ARRAY): ARRAY

-- Queries
layer_count: INTEGER
get_layer (index: INTEGER): LAYER
```

### DENSE_LAYER

```eiffel
make (input_size, output_size: INTEGER)
forward (input: ARRAY): ARRAY
backward (gradient: ARRAY): ARRAY
update_weights (learning_rate: REAL_64)
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

Backward pass computes:
1. Input gradient: `∂L/∂input = weights^T @ ∂L/∂output`
2. Weight gradient: `∂L/∂weights = ∂L/∂output @ input^T`
3. Bias gradient: `∂L/∂bias = ∂L/∂output`

### Weight Updates

Stochastic gradient descent: `w := w - learning_rate * gradient`

### Activation Functions

- **Sigmoid**: σ(z) = 1/(1 + exp(-z))
- **ReLU**: max(0, z)
- **Tanh**: (exp(z) - exp(-z)) / (exp(z) + exp(-z))

## Performance Considerations

- **Gradient Computation**: O(layer_size²) per layer
- **Training**: O(epochs * samples * network_size²)
- **Memory**: O(weights + activations) for full network

## Future Enhancements

- [ ] Batch processing
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
