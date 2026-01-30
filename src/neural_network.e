note
	description: "Neural network orchestrator with layer management and training"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class NEURAL_NETWORK

create
	make

feature {NONE} -- Initialization

	make
			-- Initialize empty network.
		do
			create layers.make (0)
			create {ARRAYED_LIST [REAL_64]} loss_history.make (0)
			is_compiled := False
		ensure
			empty: layers.is_empty
			not_compiled: not is_compiled
		end

feature -- Configuration

	add_layer (a_layer: LAYER)
			-- Add layer to network.
		require
			layer_not_void: a_layer /= Void
		do
			layers.extend (a_layer)
		ensure
			added: layers.count = old layers.count + 1
			last_is_new: layers.last = a_layer
		end

	compile (a_learning_rate: REAL_64)
			-- Compile network for training.
		require
			positive_rate: a_learning_rate > 0.0
		do
			learning_rate := a_learning_rate
			is_compiled := True
		ensure
			compiled: is_compiled
			rate_set: learning_rate = a_learning_rate
		end

feature -- Training

	fit (a_x_train: ARRAY [ARRAY [REAL_64]];
		 a_y_train: ARRAY [ARRAY [REAL_64]];
		 a_epochs: INTEGER): TRAINING_RESULT
			-- Train network using gradient descent.
		require
			data_not_empty: a_x_train.count > 0
			same_count: a_x_train.count = a_y_train.count
			positive_epochs: a_epochs > 0
		local
			l_epoch: INTEGER
			l_sample: INTEGER
			l_output: ARRAY [REAL_64]
			l_error: ARRAY [REAL_64]
			l_loss: REAL_64
		do
			create {TRAINING_RESULT} Result.make

			from l_epoch := 1
			until l_epoch > a_epochs
			loop
				l_loss := 0.0

				-- Train on each sample
				from l_sample := 1
				until l_sample > a_x_train.count
				loop
					-- Forward pass
					l_output := forward_pass (a_x_train [l_sample])

					-- Compute loss (MSE)
					l_error := compute_error (l_output, a_y_train [l_sample])
					l_loss := l_loss + mean_squared_error (l_error)

					-- Backward pass
					backward_pass (l_error)

					-- Update weights
					update_weights

					l_sample := l_sample + 1
				end

				-- Record epoch loss
				l_loss := l_loss / a_x_train.count
				loss_history.extend (l_loss)
				Result.add_epoch (l_epoch, l_loss)

				l_epoch := l_epoch + 1
			end
		ensure
			result_not_void: Result /= Void
			history_size: loss_history.count = a_epochs
		end

	predict (a_input: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Predict output for input using trained network.
		require
			input_not_void: a_input /= Void
		do
			Result := forward_pass (a_input)
		ensure
			result_not_void: Result /= Void
		end

feature -- Queries

	layer_count: INTEGER
			-- Number of layers in network.
		do
			Result := layers.count
		end

	get_layer (a_index: INTEGER): LAYER
			-- Get layer at specified index.
		require
			valid_index: a_index >= 1
		do
			Result := layers [a_index]
		ensure
			result_not_void: Result /= Void
		end

feature {NONE} -- Implementation

	forward_pass (a_input: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Propagate input through all layers.
		local
			l_activation: ARRAY [REAL_64]
			l_i: INTEGER
		do
			l_activation := a_input
			from l_i := 1
			until l_i > layers.count
			loop
				l_activation := layers [l_i].forward (l_activation)
				l_i := l_i + 1
			end
			Result := l_activation
		end

	backward_pass (a_output_error: ARRAY [REAL_64])
			-- Backpropagate error through all layers in reverse.
		local
			l_gradient: ARRAY [REAL_64]
			l_i: INTEGER
		do
			l_gradient := a_output_error

			-- Backpropagate through layers in reverse order
			from l_i := layers.count
			until l_i < 1
			loop
				l_gradient := layers [l_i].backward (l_gradient)
				l_i := l_i - 1
			end
		end

	update_weights
			-- Update all layer weights.
		local
			l_i: INTEGER
		do
			from l_i := 1
			until l_i > layers.count
			loop
				if layers [l_i].has_weights then
					layers [l_i].update_weights (learning_rate)
				end
				l_i := l_i + 1
			end
		end

	compute_error (a_output, a_target: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Compute element-wise error (output - target).
		local
			l_i: INTEGER
		do
			create Result.make_filled (0.0, 1, a_output.count)
			from l_i := 1
			until l_i > a_output.count
			loop
				Result [l_i] := a_output [l_i] - a_target [l_i]
				l_i := l_i + 1
			end
		end

	mean_squared_error (a_error: ARRAY [REAL_64]): REAL_64
			-- Compute mean squared error.
		local
			l_i: INTEGER
		do
			Result := 0.0
			from l_i := 1
			until l_i > a_error.count
			loop
				Result := Result + a_error [l_i] * a_error [l_i]
				l_i := l_i + 1
			end
			Result := Result / a_error.count
		end

	layers: ARRAYED_LIST [LAYER]
			-- Layers in network.

	learning_rate: REAL_64
			-- Learning rate for gradient descent.

	is_compiled: BOOLEAN
			-- Has network been compiled?

	loss_history: LIST [REAL_64]
			-- Loss values at each epoch.

invariant
	layers_not_void: layers /= Void
	learning_rate_positive: is_compiled implies learning_rate > 0.0

end
