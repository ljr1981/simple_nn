note
	description: "Integration test: XOR problem (non-linearly separable classification)"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class TEST_XOR_PROBLEM

inherit
	NN_TEST_SUPPORT

feature -- Tests

	test_xor_learning
			-- A seeded 2-8-4-1 sigmoid network learns XOR with per-sample SGD.
		local
			l_network: NEURAL_NETWORK
			l_result: TRAINING_RESULT
		do
			create l_network.make
			l_network.add_layer (create {DENSE_LAYER}.make_seeded (2, 8, 101))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (8))
			l_network.add_layer (create {DENSE_LAYER}.make_seeded (8, 4, 202))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (4))
			l_network.add_layer (create {DENSE_LAYER}.make_seeded (4, 1, 303))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (1))
			l_network.compile (0.5)

			l_result := l_network.fit (xor_inputs, xor_targets, 5000)

			print ("XOR Learning Test (2-8-4-1 sigmoid, seeds 101/202/303, SGD, rate 0.5, 5000 epochs):%N")
			assert_learned_xor (l_network, l_result)
		end

	test_xor_learning_with_batches
			-- A seeded 2-8-1 sigmoid network learns XOR with full-batch gradient descent
			-- (all four samples per update), which needs batch gradients to accumulate.
		local
			l_network: NEURAL_NETWORK
			l_result: TRAINING_RESULT
		do
			create l_network.make
			l_network.add_layer (create {DENSE_LAYER}.make_seeded (2, 8, 404))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (8))
			l_network.add_layer (create {DENSE_LAYER}.make_seeded (8, 1, 505))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (1))
			l_network.compile (1.0)

			l_result := l_network.fit_with_batch_size (xor_inputs, xor_targets, 5000, 4)

			print ("XOR Full-Batch Test (2-8-1 sigmoid, seeds 404/505, batch 4, rate 1.0, 5000 epochs):%N")
			assert ("one loss per epoch", l_result.epoch_count = 5000)
			assert_learned_xor (l_network, l_result)
		end

feature {NONE} -- Implementation

	Loss_threshold: REAL_64 = 0.01
			-- Final mean squared error must fall below this.

	Margin: REAL_64 = 0.2
			-- Every prediction must be within this of its 0/1 target.

	assert_learned_xor (a_network: NEURAL_NETWORK; a_result: TRAINING_RESULT)
			-- Report `a_network' on XOR and fail unless every prediction is within
			-- `Margin' of its target and the final loss is below `Loss_threshold'.
		local
			l_inputs, l_targets: ARRAY [ARRAY [REAL_64]]
			l_i: INTEGER
			l_prediction: REAL_64
		do
			l_inputs := xor_inputs
			l_targets := xor_targets
			print ("  Initial loss: " + format_real (a_result.initial_loss) + "%N")
			print ("  Final loss:   " + format_real (a_result.final_loss) + " (must be < " + Loss_threshold.out + ")%N")
			from l_i := 1 until l_i > 4 loop
				l_prediction := a_network.predict (l_inputs [l_i]) [1]
				print ("    XOR(" + l_inputs [l_i] [1].truncated_to_integer.out + "," + l_inputs [l_i] [2].truncated_to_integer.out
					+ ") = " + format_real (l_prediction) + " (target " + l_targets [l_i] [1].truncated_to_integer.out + ")%N")
				assert ("XOR sample " + l_i.out + " within margin of its target",
					(l_prediction - l_targets [l_i] [1]).abs < Margin)
				l_i := l_i + 1
			end
			assert ("final loss below threshold", a_result.final_loss < Loss_threshold)
			assert ("loss fell", a_result.final_loss < a_result.initial_loss)
		end

end
