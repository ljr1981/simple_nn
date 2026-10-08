note
	description: "Fully connected (dense) neural network layer"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class DENSE_LAYER

inherit
	LAYER

create
	make,
	make_seeded

feature {NONE} -- Initialization

	make (a_input_size, a_output_size: INTEGER)
			-- Create dense layer with Glorot (Xavier) uniform weights drawn
			-- from a seed of its own (each `make' in a thread takes the next seed).
		require
			positive_input: a_input_size > 0
			positive_output: a_output_size > 0
		do
			make_seeded (a_input_size, a_output_size, next_default_seed)
		ensure
			sizes_set: input_size = a_input_size and output_size = a_output_size
			no_gradients: accumulated_samples = 0
		end

	make_seeded (a_input_size, a_output_size, a_seed: INTEGER)
			-- Create dense layer with Glorot (Xavier) uniform weights in
			-- [-limit, limit], limit = sqrt (6 / (fan_in + fan_out)), drawn
			-- from a RANDOM seeded with `a_seed' (same seed, same weights).
		require
			positive_input: a_input_size > 0
			positive_output: a_output_size > 0
			seed_positive: a_seed > 0
				-- RANDOM is multiplicative: seed 0 yields 0 forever, so every weight would be equal.
		local
			l_i, l_j: INTEGER
			l_random: RANDOM
			l_limit: REAL_64
			l_math: SIMPLE_MATH
		do
			n_inputs := a_input_size
			n_outputs := a_output_size
			seed := a_seed

			create l_math.make
			l_limit := l_math.sqrt (6.0 / (a_input_size + a_output_size).to_double)

			create weights.make_filled (0.0, n_outputs, n_inputs)
			create l_random.set_seed (a_seed)
			l_random.start
			from l_i := 1 until l_i > n_outputs loop
				from l_j := 1 until l_j > n_inputs loop
					weights.put ((2.0 * l_random.double_item - 1.0) * l_limit, l_i, l_j)
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
			sizes_set: input_size = a_input_size and output_size = a_output_size
			seed_set: seed = a_seed
			bias_initialized: bias.count = a_output_size
			no_gradients: accumulated_samples = 0
			no_input_yet: not has_cached_input
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

feature -- Access

	seed: INTEGER
			-- Seed the initial weights were drawn from.

	weight (a_row, a_column: INTEGER): REAL_64
			-- Weight from input `a_column' to output `a_row'.
		require
			valid_row: a_row >= 1 and a_row <= output_size
			valid_column: a_column >= 1 and a_column <= input_size
		do
			Result := weights.item (a_row, a_column)
		end

	bias_value (a_row: INTEGER): REAL_64
			-- Bias of output `a_row'.
		require
			valid_row: a_row >= 1 and a_row <= output_size
		do
			Result := bias [a_row]
		end

	weight_gradient (a_row, a_column: INTEGER): REAL_64
			-- Sum of dLoss/d`weight (a_row, a_column)' over the `accumulated_samples' samples.
		require
			valid_row: a_row >= 1 and a_row <= output_size
			valid_column: a_column >= 1 and a_column <= input_size
		do
			Result := weight_gradients.item (a_row, a_column)
		end

	bias_gradient (a_row: INTEGER): REAL_64
			-- Sum of dLoss/d`bias_value (a_row)' over the `accumulated_samples' samples.
		require
			valid_row: a_row >= 1 and a_row <= output_size
		do
			Result := bias_gradients [a_row]
		end

	accumulated_samples: INTEGER
			-- Number of `backward' passes summed into the gradients since the last clear.

feature -- Status report

	has_cached_input: BOOLEAN
			-- <Precursor>
		do
			Result := last_input.count = n_inputs
		end

feature -- Element change

	set_weight (a_row, a_column: INTEGER; a_value: REAL_64)
			-- Set the weight from input `a_column' to output `a_row' to `a_value'.
		require
			valid_row: a_row >= 1 and a_row <= output_size
			valid_column: a_column >= 1 and a_column <= input_size
			not_nan: not a_value.is_nan
		do
			weights.put (a_value, a_row, a_column)
		ensure
			weight_set: weight (a_row, a_column) = a_value
		end

	set_bias (a_row: INTEGER; a_value: REAL_64)
			-- Set the bias of output `a_row' to `a_value'.
		require
			valid_row: a_row >= 1 and a_row <= output_size
			not_nan: not a_value.is_nan
		do
			bias [a_row] := a_value
		ensure
			bias_set: bias_value (a_row) = a_value
		end

feature -- Forward/Backward

	forward (a_input: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Compute forward pass: output = weights @ input + bias
		local
			l_i, l_j: INTEGER
			l_sum: REAL_64
		do
				-- Own copy: a caller reusing its array must not change what `backward' sees.
			last_input := a_input.twin
			create Result.make_filled (0.0, 1, n_outputs)

			from l_i := 1 until l_i > n_outputs loop
				l_sum := bias [l_i]
				from l_j := 1 until l_j > n_inputs loop
					l_sum := l_sum + weights.item (l_i, l_j) * last_input [l_j]
					l_j := l_j + 1
				end
				Result [l_i] := l_sum
				l_i := l_i + 1
			end
		end

	backward (a_output_gradient: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Add this sample's weight and bias gradients to the accumulators.
			-- Returns gradient with respect to input: weights^T @ output_gradient.
		local
			l_i, l_j: INTEGER
			l_gradient: REAL_64
		do
			create Result.make_filled (0.0, 1, n_inputs)

			from l_i := 1 until l_i > n_outputs loop
				l_gradient := a_output_gradient [l_i]
				from l_j := 1 until l_j > n_inputs loop
					Result [l_j] := Result [l_j] + weights.item (l_i, l_j) * l_gradient
					weight_gradients.put (weight_gradients.item (l_i, l_j) + l_gradient * last_input [l_j], l_i, l_j)
					l_j := l_j + 1
				end
				bias_gradients [l_i] := bias_gradients [l_i] + l_gradient
				l_i := l_i + 1
			end
			accumulated_samples := accumulated_samples + 1
		ensure then
			one_more_sample: accumulated_samples = old accumulated_samples + 1
		end

feature -- Weights

	has_weights: BOOLEAN = True

	update_weights (a_learning_rate: REAL_64)
			-- Step weights and bias against the accumulated gradients
			-- (w := w - a_learning_rate * gradient), then clear the gradients.
		local
			l_i, l_j: INTEGER
		do
			from l_i := 1 until l_i > n_outputs loop
				from l_j := 1 until l_j > n_inputs loop
					weights.put (
						weights.item (l_i, l_j) - a_learning_rate * weight_gradients.item (l_i, l_j),
						l_i, l_j
					)
					l_j := l_j + 1
				end
				bias [l_i] := bias [l_i] - a_learning_rate * bias_gradients [l_i]
				l_i := l_i + 1
			end
			clear_gradients
		ensure then
			gradients_cleared: accumulated_samples = 0
		end

	clear_gradients
			-- <Precursor>
		local
			l_i, l_j: INTEGER
		do
			from l_i := 1 until l_i > n_outputs loop
				from l_j := 1 until l_j > n_inputs loop
					weight_gradients.put (0.0, l_i, l_j)
					l_j := l_j + 1
				end
				bias_gradients [l_i] := 0.0
				l_i := l_i + 1
			end
			accumulated_samples := 0
		ensure then
			cleared: accumulated_samples = 0
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
			-- Copy of the input from the most recent forward pass (for backward).

	next_default_seed: INTEGER
			-- A fresh positive seed for `make', so that no two layers share a weight sequence.
		do
			Result := default_seed_source.item
			default_seed_source.forth
			if Result <= 0 then
				Result := 1
			end
		ensure
			positive: Result > 0
		end

	default_seed_source: RANDOM
			-- Per-thread generator of seeds for `make'.
		once
			create Result.make
		end

invariant
	sizes_positive: n_inputs > 0 and n_outputs > 0
	weights_shape: weights.height = n_outputs and weights.width = n_inputs
	bias_count: bias.count = n_outputs
	gradient_shapes: weight_gradients.height = n_outputs and weight_gradients.width = n_inputs and bias_gradients.count = n_outputs
	seed_positive: seed > 0
	samples_non_negative: accumulated_samples >= 0

end
