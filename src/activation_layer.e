note
	description: "Activation function layer for neural networks"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class ACTIVATION_LAYER

inherit
	LAYER

create
	make_sigmoid,
	make_relu,
	make_tanh

feature {NONE} -- Initialization

	make_sigmoid (a_size: INTEGER)
			-- Create sigmoid activation layer.
		require
			positive_size: a_size > 0
		do
			size := a_size
			activation_type := Sigmoid_type
			create activation_functions.make
			create last_input.make_empty
		ensure
			size_set: size = a_size
			type_sigmoid: activation_type = Sigmoid_type
		end

	make_relu (a_size: INTEGER)
			-- Create ReLU activation layer.
		require
			positive_size: a_size > 0
		do
			size := a_size
			activation_type := Relu_type
			create activation_functions.make
			create last_input.make_empty
		ensure
			size_set: size = a_size
			type_relu: activation_type = Relu_type
		end

	make_tanh (a_size: INTEGER)
			-- Create tanh activation layer.
		require
			positive_size: a_size > 0
		do
			size := a_size
			activation_type := Tanh_type
			create activation_functions.make
			create last_input.make_empty
		ensure
			size_set: size = a_size
			type_tanh: activation_type = Tanh_type
		end

feature -- Dimensions

	input_size: INTEGER
		do
			Result := size
		end

	output_size: INTEGER
		do
			Result := size
		end

feature -- Forward/Backward

	forward (a_input: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Apply activation function element-wise.
		local
			l_i: INTEGER
		do
			last_input := a_input
			create Result.make_filled (0.0, 1, size)

			from l_i := 1 until l_i > size loop
				inspect activation_type
				when Sigmoid_type then
					Result [l_i] := activation_functions.sigmoid (a_input [l_i])
				when Relu_type then
					Result [l_i] := activation_functions.relu (a_input [l_i])
				when Tanh_type then
					Result [l_i] := activation_functions.tanh (a_input [l_i])
				end
				l_i := l_i + 1
			end
		end

	backward (a_output_gradient: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Compute gradient of activation function and multiply by output gradient.
		local
			l_i: INTEGER
			l_derivative: REAL_64
		do
			create Result.make_filled (0.0, 1, size)

			from l_i := 1 until l_i > size loop
				inspect activation_type
				when Sigmoid_type then
					l_derivative := activation_functions.sigmoid_derivative (last_input [l_i])
				when Relu_type then
					l_derivative := activation_functions.relu_derivative (last_input [l_i])
				when Tanh_type then
					l_derivative := activation_functions.tanh_derivative (last_input [l_i])
				end
				Result [l_i] := a_output_gradient [l_i] * l_derivative
				l_i := l_i + 1
			end
		end

feature -- Weights

	has_weights: BOOLEAN = False

	update_weights (a_learning_rate: REAL_64)
			-- No weights to update for activation layer.
		do
			-- Activation functions have no learnable parameters
		end

feature {NONE} -- Implementation

	size: INTEGER
			-- Number of units in layer.

	activation_type: INTEGER
			-- Type of activation function.

	last_input: ARRAY [REAL_64]
			-- Input from most recent forward pass (for backward).

	activation_functions: ACTIVATION_FUNCTIONS
			-- Activation function implementations.

	Sigmoid_type: INTEGER = 1
	Relu_type: INTEGER = 2
	Tanh_type: INTEGER = 3

invariant
	size_positive: size > 0
	valid_type: activation_type >= 1 and activation_type <= 3
	functions_not_void: activation_functions /= Void

end
