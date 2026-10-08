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
			-- Train network using stochastic gradient descent (sample-by-sample).
		require
			compiled: is_compiled
			has_layers: layer_count > 0
			data_not_empty: a_x_train.count > 0
			same_count: a_x_train.count = a_y_train.count
			positive_epochs: a_epochs > 0
		do
			Result := fit_with_batch_size (a_x_train, a_y_train, a_epochs, 1)
		ensure
			result_not_void: Result /= Void
		end

	fit_with_batch_size (a_x_train: ARRAY [ARRAY [REAL_64]];
						 a_y_train: ARRAY [ARRAY [REAL_64]];
						 a_epochs: INTEGER;
						 a_batch_size: INTEGER): TRAINING_RESULT
			-- Train network using mini-batch gradient descent.
			-- Each batch's per-sample gradients are summed, then applied once
			-- with rate `learning_rate' / batch length (the batch-mean gradient).
			-- a_batch_size = 1: Stochastic gradient descent (SGD)
			-- a_batch_size = dataset_size: Batch gradient descent
			-- a_batch_size > 1: Mini-batch gradient descent
		require
			compiled: is_compiled
			has_layers: layer_count > 0
			data_not_empty: a_x_train.count > 0
			same_count: a_x_train.count = a_y_train.count
			positive_epochs: a_epochs > 0
			positive_batch_size: a_batch_size > 0
		local
			l_epoch: INTEGER
			l_sample: INTEGER
			l_batch_start: INTEGER
			l_batch_end: INTEGER
			l_batch_sample: INTEGER
			l_output: ARRAY [REAL_64]
			l_loss: REAL_64
			l_batch_loss: REAL_64
			l_batch_count: INTEGER
		do
			create {TRAINING_RESULT} Result.make
			clear_gradients

			from l_epoch := 1
			until l_epoch > a_epochs
			loop
				l_loss := 0.0
				l_batch_count := 0

				-- Process batches
				from l_sample := 1
				until l_sample > a_x_train.count
				loop
					-- Define batch boundaries
					l_batch_start := l_sample
					l_batch_end := (l_sample + a_batch_size - 1).min (a_x_train.count)
					l_batch_loss := 0.0

					-- Accumulate gradients for batch
					from l_batch_sample := l_batch_start
					until l_batch_sample > l_batch_end
					loop
						-- Forward pass
						l_output := forward_pass (a_x_train [l_batch_sample])

						-- Compute loss (MSE)
						l_batch_loss := l_batch_loss + mean_squared_error (compute_error (l_output, a_y_train [l_batch_sample]))

						-- Backward pass: gradients add up in the dense layers
						backward_pass (loss_gradient (l_output, a_y_train [l_batch_sample]))

						l_batch_sample := l_batch_sample + 1
					end

					-- Update weights once per batch, with the batch-mean gradient
					update_weights (learning_rate / (l_batch_end - l_batch_start + 1))

					-- Average batch loss
					l_batch_loss := l_batch_loss / (l_batch_end - l_batch_start + 1)
					l_loss := l_loss + l_batch_loss
					l_batch_count := l_batch_count + 1

					-- Move to next batch
					l_sample := l_batch_end + 1
				end

				-- Record epoch loss
				l_loss := l_loss / l_batch_count
				loss_history.extend (l_loss)
				Result.add_epoch (l_epoch, l_loss)

				l_epoch := l_epoch + 1
			end
		ensure
			result_not_void: Result /= Void
			history_size: loss_history.count = a_epochs
		end

	compute_gradients (a_input, a_target: ARRAY [REAL_64])
			-- Clear all gradients, then backpropagate one sample, so that each
			-- dense layer's `weight_gradient' and `bias_gradient' hold the
			-- derivatives of `loss (a_input, a_target)'. Weights are not changed.
		require
			has_layers: layer_count > 0
			input_size_matches: a_input.count = input_size
			target_size_matches: a_target.count = output_size
		do
			clear_gradients
			backward_pass (loss_gradient (forward_pass (a_input), a_target))
		end

	predict (a_input: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Predict output for input using trained network.
		require
			input_not_void: a_input /= Void
			has_layers: layer_count > 0
			input_size_matches: a_input.count = input_size
		do
			Result := forward_pass (a_input)
		ensure
			result_not_void: Result /= Void
		end

feature -- Measurement

	loss (a_input, a_target: ARRAY [REAL_64]): REAL_64
			-- Mean squared error of the prediction for `a_input' against `a_target':
			-- (1/n) * sum ((output - target)^2).
		require
			has_layers: layer_count > 0
			input_size_matches: a_input.count = input_size
			target_size_matches: a_target.count = output_size
		do
			Result := mean_squared_error (compute_error (forward_pass (a_input), a_target))
		ensure
			non_negative: Result >= 0.0
		end

feature -- Queries

	learning_rate: REAL_64
			-- Learning rate for gradient descent.

	is_compiled: BOOLEAN
			-- Has network been compiled?

	input_size: INTEGER
			-- Width of the input the first layer takes.
		require
			has_layers: layer_count > 0
		do
			Result := layers.first.input_size
		end

	output_size: INTEGER
			-- Width of the output the last layer produces.
		require
			has_layers: layer_count > 0
		do
			Result := layers.last.output_size
		end

	layer_count: INTEGER
			-- Number of layers in network.
		do
			Result := layers.count
		end

	get_layer (a_index: INTEGER): LAYER
			-- Get layer at specified index.
		require
			valid_index: a_index >= 1 and a_index <= layer_count
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

	update_weights (a_rate: REAL_64)
			-- Step every weighted layer against its accumulated gradients with rate `a_rate'.
		require
			positive_rate: a_rate > 0.0
		local
			l_i: INTEGER
		do
			from l_i := 1
			until l_i > layers.count
			loop
				if layers [l_i].has_weights then
					layers [l_i].update_weights (a_rate)
				end
				l_i := l_i + 1
			end
		end

	clear_gradients
			-- Discard the gradients accumulated in every layer.
		local
			l_i: INTEGER
		do
			from l_i := 1
			until l_i > layers.count
			loop
				layers [l_i].clear_gradients
				l_i := l_i + 1
			end
		end

	loss_gradient (a_output, a_target: ARRAY [REAL_64]): ARRAY [REAL_64]
			-- Gradient of `mean_squared_error' with respect to the output:
			-- d/d output_i of (1/n) sum (output - target)^2 = 2 (output_i - target_i) / n.
		require
			same_count: a_output.count = a_target.count
			not_empty: a_output.count > 0
		local
			l_i: INTEGER
		do
			create Result.make_filled (0.0, 1, a_output.count)
			from l_i := 1
			until l_i > a_output.count
			loop
				Result [l_i] := 2.0 * (a_output [l_i] - a_target [l_i]) / a_output.count
				l_i := l_i + 1
			end
		ensure
			same_count: Result.count = a_output.count
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

	loss_history: LIST [REAL_64]
			-- Loss values at each epoch.

invariant
	layers_not_void: layers /= Void
	learning_rate_positive: is_compiled implies learning_rate > 0.0

end
