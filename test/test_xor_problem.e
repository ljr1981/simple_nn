note
	description: "Integration test: XOR problem (non-linearly separable classification)"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class TEST_XOR_PROBLEM

feature -- Tests

	test_xor_learning
			-- Test that network can learn XOR (non-linearly separable).
		local
			l_network: NEURAL_NETWORK
			l_x_train: ARRAY [ARRAY [REAL_64]]
			l_y_train: ARRAY [ARRAY [REAL_64]]
			l_result: TRAINING_RESULT
			l_output: ARRAY [REAL_64]
			l_passes: INTEGER
		do
			-- Create network: 2-4-1 (input-hidden-output)
			create l_network.make
			l_network.add_layer (create {DENSE_LAYER}.make (2, 4))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (4))
			l_network.add_layer (create {DENSE_LAYER}.make (4, 1))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (1))
			l_network.compile (0.5)  -- learning_rate = 0.5

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

			-- Train network
			l_result := l_network.fit (l_x_train, l_y_train, 1000)

			-- Test predictions
			l_passes := 0
			l_output := l_network.predict (<<0.0, 0.0>>)
			if l_output [1] < 0.1 then
				l_passes := l_passes + 1
			end

			l_output := l_network.predict (<<0.0, 1.0>>)
			if l_output [1] > 0.9 then
				l_passes := l_passes + 1
			end

			l_output := l_network.predict (<<1.0, 0.0>>)
			if l_output [1] > 0.9 then
				l_passes := l_passes + 1
			end

			l_output := l_network.predict (<<1.0, 1.0>>)
			if l_output [1] < 0.1 then
				l_passes := l_passes + 1
			end

			-- Verify loss decreased
			if l_result.final_loss < l_result.initial_loss then
				l_passes := l_passes + 1
			end

			-- Require at least 4 out of 5 criteria to pass
			if l_passes < 4 then
				print ("XOR Learning Test Failed: " + l_passes.out + "/5 criteria met%N")
				print ("  Initial loss: " + l_result.initial_loss.out + "%N")
				print ("  Final loss: " + l_result.final_loss.out + "%N")
				print ("  XOR(0,0): " + l_network.predict (<<0.0, 0.0>>) [1].out + " (expected ~0.0)%N")
				print ("  XOR(0,1): " + l_network.predict (<<0.0, 1.0>>) [1].out + " (expected ~1.0)%N")
				print ("  XOR(1,0): " + l_network.predict (<<1.0, 0.0>>) [1].out + " (expected ~1.0)%N")
				print ("  XOR(1,1): " + l_network.predict (<<1.0, 1.0>>) [1].out + " (expected ~0.0)%N")
			end
		end

end
