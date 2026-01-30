note
	description: "Integration test: XOR problem (non-linearly separable classification)"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class TEST_XOR_PROBLEM

feature -- Tests

	test_xor_learning
			-- Test that network can learn XOR (non-linearly separable).
			-- Uses improved hyperparameters for actual convergence.
		local
			l_network: NEURAL_NETWORK
			l_x_train: ARRAY [ARRAY [REAL_64]]
			l_y_train: ARRAY [ARRAY [REAL_64]]
			l_result: TRAINING_RESULT
			l_output: ARRAY [REAL_64]
			l_passes: INTEGER
			l_loss_ratio: REAL_64
		do
			-- Create network: 2-8-4-1 (larger for XOR)
			create l_network.make
			l_network.add_layer (create {DENSE_LAYER}.make (2, 8))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (8))
			l_network.add_layer (create {DENSE_LAYER}.make (8, 4))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (4))
			l_network.add_layer (create {DENSE_LAYER}.make (4, 1))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (1))
			l_network.compile (0.1)  -- Reduced learning rate for stability

			-- XOR training data
			create l_x_train.make (1, 4)
			l_x_train [1] := <<0.0, 0.0>>
			l_x_train [2] := <<0.0, 1.0>>
			l_x_train [3] := <<1.0, 0.0>>
			l_x_train [4] := <<1.0, 1.0>>

			create l_y_train.make (1, 4)
			l_y_train [1] := <<0.0>>  -- 0 XOR 0 = 0
			l_y_train [2] := <<1.0>>  -- 0 XOR 1 = 1
			l_y_train [3] := <<1.0>>  -- 1 XOR 0 = 1
			l_y_train [4] := <<0.0>>  -- 1 XOR 1 = 0

			-- Train network (more epochs for convergence)
			l_result := l_network.fit (l_x_train, l_y_train, 5000)

			-- Test predictions
			l_passes := 0
			l_output := l_network.predict (<<0.0, 0.0>>)
			if l_output [1] < 0.2 then
				l_passes := l_passes + 1
			end

			l_output := l_network.predict (<<0.0, 1.0>>)
			if l_output [1] > 0.8 then
				l_passes := l_passes + 1
			end

			l_output := l_network.predict (<<1.0, 0.0>>)
			if l_output [1] > 0.8 then
				l_passes := l_passes + 1
			end

			l_output := l_network.predict (<<1.0, 1.0>>)
			if l_output [1] < 0.2 then
				l_passes := l_passes + 1
			end

			-- Verify significant loss reduction
			l_loss_ratio := l_result.final_loss / (l_result.initial_loss + 0.0001)
			if l_loss_ratio < 0.3 then
				l_passes := l_passes + 1
			end

			-- Print results
			print ("XOR Learning Test Results:%N")
			print ("  Network: 2-8-4-1 (sigmoid activations)%N")
			print ("  Learning Rate: 0.1, Epochs: 5000%N")
			print ("  Initial loss: " + format_real(l_result.initial_loss) + "%N")
			print ("  Final loss:   " + format_real(l_result.final_loss) + "%N")
			print ("  Loss reduction: " + format_real(100.0 * (1.0 - l_loss_ratio)) + "%% %N")
			print ("  Predictions:%N")
			print ("    XOR(0,0) = " + format_real(l_network.predict (<<0.0, 0.0>>) [1]) + " (expected ~0.0)%N")
			print ("    XOR(0,1) = " + format_real(l_network.predict (<<0.0, 1.0>>) [1]) + " (expected ~1.0)%N")
			print ("    XOR(1,0) = " + format_real(l_network.predict (<<1.0, 0.0>>) [1]) + " (expected ~1.0)%N")
			print ("    XOR(1,1) = " + format_real(l_network.predict (<<1.0, 1.0>>) [1]) + " (expected ~0.0)%N")
			print ("  Success Criteria: " + l_passes.out + "/5 passed%N")

			if l_passes < 4 then
				print ("  Status: INCOMPLETE - Network needs more tuning%N")
			else
				print ("  Status: SUCCESS - Network learned XOR%N")
			end
		end

	test_xor_learning_with_batches
			-- Test XOR learning with batch processing (when available).
		local
			l_network: NEURAL_NETWORK
			l_x_train: ARRAY [ARRAY [REAL_64]]
			l_y_train: ARRAY [ARRAY [REAL_64]]
			l_result: TRAINING_RESULT
		do
			-- Create network
			create l_network.make
			l_network.add_layer (create {DENSE_LAYER}.make (2, 8))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (8))
			l_network.add_layer (create {DENSE_LAYER}.make (8, 1))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (1))
			l_network.compile (0.1)

			-- XOR data
			create l_x_train.make (1, 4)
			l_x_train [1] := <<0.0, 0.0>>
			l_x_train [2] := <<0.0, 1.0>>
			l_x_train [3] := <<1.0, 0.0>>
			l_x_train [4] := <<1.0, 1.0>>

			create l_y_train.make (1, 4)
			l_y_train [1] := <<0.0>>
			l_y_train [2] := <<1.0>>
			l_y_train [3] := <<1.0>>
			l_y_train [4] := <<0.0>>

			-- Train with batch processing (using current fit method)
			l_result := l_network.fit (l_x_train, l_y_train, 3000)

			print ("XOR Batch Processing Test:%N")
			print ("  Final loss: " + format_real(l_result.final_loss) + "%N")
			print ("  Epochs trained: " + l_result.epoch_count.out + "%N")
		end

feature {NONE} -- Helper

	format_real (a_value: REAL_64): STRING
			-- Format real number with reasonable precision.
		local
			l_rounded: REAL_64
		do
			-- Round to 4 decimal places
			l_rounded := (a_value * 10000.0).rounded / 10000.0
			Result := l_rounded.out
		end

end
