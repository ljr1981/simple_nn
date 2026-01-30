note
	description: "Simple neural network library facade"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class SIMPLE_NN

create
	make

feature {NONE} -- Initialization

	make
			-- Initialize neural network library.
		do
		end

feature -- Factory Methods

	new_network: NEURAL_NETWORK
			-- Create new neural network.
		do
			create Result.make
		end

	new_dense_layer (a_input_size, a_output_size: INTEGER): DENSE_LAYER
			-- Create new dense (fully connected) layer.
		require
			positive_input: a_input_size > 0
			positive_output: a_output_size > 0
		do
			create Result.make (a_input_size, a_output_size)
		end

	new_sigmoid_layer (a_size: INTEGER): ACTIVATION_LAYER
			-- Create new sigmoid activation layer.
		require
			positive_size: a_size > 0
		do
			create Result.make_sigmoid (a_size)
		end

	new_relu_layer (a_size: INTEGER): ACTIVATION_LAYER
			-- Create new ReLU activation layer.
		require
			positive_size: a_size > 0
		do
			create Result.make_relu (a_size)
		end

	new_tanh_layer (a_size: INTEGER): ACTIVATION_LAYER
			-- Create new tanh activation layer.
		require
			positive_size: a_size > 0
		do
			create Result.make_tanh (a_size)
		end

end
