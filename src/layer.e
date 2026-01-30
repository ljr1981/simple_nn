note
	description: "Deferred base class for neural network layers"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

deferred class LAYER

feature -- Dimensions

	input_size: INTEGER
			-- Number of input units.
		deferred
		ensure
			positive: Result > 0
		end

	output_size: INTEGER
			-- Number of output units.
		deferred
		ensure
			positive: Result > 0
		end

feature -- Forward/Backward Propagation

	forward (a_input: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Compute forward pass through layer.
		require
			input_not_void: a_input /= Void
			correct_size: a_input.count = input_size
		deferred
		ensure
			result_not_void: Result /= Void
			correct_output_size: Result.count = output_size
		end

	backward (a_output_gradient: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Compute backward pass through layer.
			-- Returns gradient with respect to input.
		require
			gradient_not_void: a_output_gradient /= Void
			correct_size: a_output_gradient.count = output_size
		deferred
		ensure
			result_not_void: Result /= Void
			correct_input_size: Result.count = input_size
		end

feature -- Weight Management

	has_weights: BOOLEAN
			-- Does this layer have trainable weights?
		deferred
		end

	update_weights (a_learning_rate: REAL_64)
			-- Update layer weights using accumulated gradients.
		require
			has_weights: has_weights
			positive_rate: a_learning_rate > 0.0
		deferred
		end

end
