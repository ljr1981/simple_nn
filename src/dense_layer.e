note
	description: "Fully connected (dense) neural network layer"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class DENSE_LAYER

inherit
	LAYER

create
	make

feature {NONE} -- Initialization

	make (a_input_size, a_output_size: INTEGER)
			-- Create dense layer with specified dimensions.
			-- Weights initialized using Xavier initialization.
		require
			positive_input: a_input_size > 0
			positive_output: a_output_size > 0
		local
			l_i, l_j: INTEGER
			l_random: RANDOM
			l_scale: REAL_64
			l_math: SIMPLE_MATH
		do
			n_inputs := a_input_size
			n_outputs := a_output_size

			-- Initialize weights with Xavier initialization
			create l_math.make
			l_scale := l_math.sqrt (2.0 / a_input_size)

			create weights.make_filled (0.0, n_outputs, n_inputs)
			create l_random.make
			from l_i := 1 until l_i > n_outputs loop
				from l_j := 1 until l_j > n_inputs loop
					weights.put (l_random.double_item * l_scale - (l_scale / 2.0), l_i, l_j)
					l_random.forth
					l_j := l_j + 1
				end
				l_i := l_i + 1
			end

			create bias.make_filled (0.0, 1, n_outputs)
			create weight_gradients.make_filled (0.0, n_outputs, n_inputs)
			create bias_gradients.make_filled (0.0, 1, n_outputs)
			create last_input.make_empty
		ensure
			sizes_set: n_inputs = a_input_size and n_outputs = a_output_size
			bias_initialized: bias.count = a_output_size
		end

feature -- Dimensions

	input_size: INTEGER
		do
			Result := n_inputs
		end

	output_size: INTEGER
		do
			Result := n_outputs
		end

feature -- Forward/Backward

	forward (a_input: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Compute forward pass: output = weights @ input + bias
		local
			l_i, l_j: INTEGER
			l_sum: REAL_64
		do
			last_input := a_input
			create Result.make_filled (0.0, 1, n_outputs)

			from l_i := 1 until l_i > n_outputs loop
				l_sum := bias [l_i]
				from l_j := 1 until l_j > n_inputs loop
					l_sum := l_sum + weights.item (l_i, l_j) * a_input [l_j]
					l_j := l_j + 1
				end
				Result [l_i] := l_sum
				l_i := l_i + 1
			end
		end

	backward (a_output_gradient: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Compute backward pass and accumulate gradients.
			-- Returns gradient with respect to input.
		local
			l_i, l_j: INTEGER
		do
			create Result.make_filled (0.0, 1, n_inputs)

			-- Compute input gradient: weights^T @ output_gradient
			from l_j := 1 until l_j > n_inputs loop
				from l_i := 1 until l_i > n_outputs loop
					Result [l_j] := Result [l_j] + weights.item (l_i, l_j) * a_output_gradient [l_i]
					l_i := l_i + 1
				end
				l_j := l_j + 1
			end

			-- Compute weight gradients: output_gradient @ input^T
			from l_i := 1 until l_i > n_outputs loop
				from l_j := 1 until l_j > n_inputs loop
					weight_gradients.put (a_output_gradient [l_i] * last_input [l_j], l_i, l_j)
					l_j := l_j + 1
				end
				l_i := l_i + 1
			end

			-- Bias gradients = output_gradient
			bias_gradients := a_output_gradient
		end

feature -- Weights

	has_weights: BOOLEAN = True

	update_weights (a_learning_rate: REAL_64)
			-- Update weights and bias using accumulated gradients.
		local
			l_i, l_j: INTEGER
		do
			-- Update weights: w -= learning_rate * gradient
			from l_i := 1 until l_i > n_outputs loop
				from l_j := 1 until l_j > n_inputs loop
					weights.put (
						weights.item (l_i, l_j) - a_learning_rate * weight_gradients.item (l_i, l_j),
						l_i, l_j
					)
					l_j := l_j + 1
				end
				l_i := l_i + 1
			end

			-- Update bias: b -= learning_rate * gradient
			from l_i := 1 until l_i > n_outputs loop
				bias [l_i] := bias [l_i] - a_learning_rate * bias_gradients [l_i]
				l_i := l_i + 1
			end
		end

feature {NONE} -- Implementation

	n_inputs, n_outputs: INTEGER
			-- Layer dimensions.

	weights: ARRAY2 [REAL_64]
			-- Weight matrix [n_outputs x n_inputs].

	bias: ARRAY [REAL_64]
			-- Bias vector [n_outputs].

	weight_gradients: ARRAY2 [REAL_64]
			-- Accumulated weight gradients.

	bias_gradients: ARRAY [REAL_64]
			-- Accumulated bias gradients.

	last_input: ARRAY [REAL_64]
			-- Input from most recent forward pass (for backward).

invariant
	sizes_positive: n_inputs > 0 and n_outputs > 0
	weights_not_void: weights /= Void
	bias_not_void: bias /= Void

end
