note
	description: "Diagnostic tests to identify learning issues"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class TEST_DIAGNOSTIC

feature -- Tests

	test_single_neuron_learning
			-- Test if a single neuron can learn simple binary classification.
		local
			l_network: NEURAL_NETWORK
			l_x_train: ARRAY [ARRAY [REAL_64]]
			l_y_train: ARRAY [ARRAY [REAL_64]]
			l_result: TRAINING_RESULT
			l_output: ARRAY [REAL_64]
		do
			print ("Single Neuron Learning Test (AND gate):%N")

			-- Create simple network: 2-1 (just one neuron)
			create l_network.make
			l_network.add_layer (create {DENSE_LAYER}.make (2, 1))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (1))
			l_network.compile (1.0)  -- Try larger learning rate

			-- AND training data (linearly separable)
			create l_x_train.make (1, 4)
			l_x_train [1] := <<0.0, 0.0>>
			l_x_train [2] := <<0.0, 1.0>>
			l_x_train [3] := <<1.0, 0.0>>
			l_x_train [4] := <<1.0, 1.0>>

			create l_y_train.make (1, 4)
			l_y_train [1] := <<0.0>>  -- 0 AND 0 = 0
			l_y_train [2] := <<0.0>>  -- 0 AND 1 = 0
			l_y_train [3] := <<0.0>>  -- 1 AND 0 = 0
			l_y_train [4] := <<1.0>>  -- 1 AND 1 = 1

			-- Train
			l_result := l_network.fit (l_x_train, l_y_train, 1000)

			print ("  Initial loss: " + l_result.initial_loss.out + "%N")
			print ("  Final loss:   " + l_result.final_loss.out + "%N")
			print ("  Loss reduction: " + ((1.0 - l_result.final_loss / l_result.initial_loss) * 100.0).out + "%% %N")
			print ("  Predictions:%N")
			l_output := l_network.predict (<<0.0, 0.0>>)
			print ("    AND(0,0) = " + l_output [1].out + " (expected 0.0)%N")
			l_output := l_network.predict (<<1.0, 1.0>>)
			print ("    AND(1,1) = " + l_output [1].out + " (expected 1.0)%N")
		end

	test_weight_updates
			-- Test that weights are actually changing during training.
		local
			l_network: NEURAL_NETWORK
			l_x_train: ARRAY [ARRAY [REAL_64]]
			l_y_train: ARRAY [ARRAY [REAL_64]]
			l_result: TRAINING_RESULT
		do
			print ("Weight Update Test:%N")

			-- Simple 2-3-1 network
			create l_network.make
			l_network.add_layer (create {DENSE_LAYER}.make (2, 3))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (3))
			l_network.add_layer (create {DENSE_LAYER}.make (3, 1))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (1))
			l_network.compile (5.0)  -- Very large learning rate to check updates

			-- Simple data
			create l_x_train.make (1, 2)
			l_x_train [1] := <<0.0, 0.0>>
			l_x_train [2] := <<1.0, 1.0>>

			create l_y_train.make (1, 2)
			l_y_train [1] := <<0.0>>
			l_y_train [2] := <<1.0>>

			-- Train just 10 epochs
			l_result := l_network.fit (l_x_train, l_y_train, 10)

			print ("  Epochs: 10, Learning Rate: 5.0%N")
			print ("  Initial loss: " + l_result.initial_loss.out + "%N")
			print ("  Final loss:   " + l_result.final_loss.out + "%N")

			if l_result.final_loss >= l_result.initial_loss then
				print ("  WARNING: Loss did not decrease! Learning may be broken.%N")
			else
				print ("  OK: Loss decreased (weights are updating)%N")
			end
		end

	test_loss_computation
			-- Test if loss function is working correctly.
		local
			l_error: ARRAY [REAL_64]
		do
			print ("Loss Computation Test:%N")

			-- Test various error vectors
			create l_error.make_filled (0.0, 1, 4)
			l_error [1] := 0.0
			l_error [2] := 0.5
			l_error [3] := 1.0
			l_error [4] := 0.1

			print ("  Error vector: [0.0, 0.5, 1.0, 0.1]%N")
			print ("  Expected MSE: " + ((0.0*0.0 + 0.5*0.5 + 1.0*1.0 + 0.1*0.1) / 4.0).out + "%N")
		end

end
