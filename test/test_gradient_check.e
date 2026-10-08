note
	description: "[
		Numerical gradient check: every weight and bias gradient that backprop
		produces must match a central finite difference of NEURAL_NETWORK.loss.
	]"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class TEST_GRADIENT_CHECK

inherit
	NN_TEST_SUPPORT

feature -- Tests

	test_gradient_check
			-- 3-4-2 network (tanh hidden, sigmoid output, two outputs so the 1/n of
			-- the MSE is exercised): all 26 parameter gradients match finite differences.
		local
			l_network: NEURAL_NETWORK
			l_input, l_target: ARRAY [REAL_64]
			l_layer_index, l_row, l_column, l_checked: INTEGER
			l_original, l_numeric, l_max_error: REAL_64
		do
			create l_network.make
			l_network.add_layer (create {DENSE_LAYER}.make_seeded (3, 4, 7))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_tanh (4))
			l_network.add_layer (create {DENSE_LAYER}.make_seeded (4, 2, 13))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (2))
			l_input := <<0.3, -0.7, 0.9>>
			l_target := <<1.0, 0.0>>

			l_network.compute_gradients (l_input, l_target)

			from l_layer_index := 1 until l_layer_index > l_network.layer_count loop
				if attached {DENSE_LAYER} l_network.get_layer (l_layer_index) as al_dense then
					assert ("one sample accumulated", al_dense.accumulated_samples = 1)
					from l_row := 1 until l_row > al_dense.output_size loop
						from l_column := 1 until l_column > al_dense.input_size loop
							l_original := al_dense.weight (l_row, l_column)
							al_dense.set_weight (l_row, l_column, l_original + Step)
							l_numeric := l_network.loss (l_input, l_target)
							al_dense.set_weight (l_row, l_column, l_original - Step)
							l_numeric := (l_numeric - l_network.loss (l_input, l_target)) / (2.0 * Step)
							al_dense.set_weight (l_row, l_column, l_original)
							l_max_error := l_max_error.max (relative_error (l_numeric, al_dense.weight_gradient (l_row, l_column)))
							l_checked := l_checked + 1
							l_column := l_column + 1
						end
						l_original := al_dense.bias_value (l_row)
						al_dense.set_bias (l_row, l_original + Step)
						l_numeric := l_network.loss (l_input, l_target)
						al_dense.set_bias (l_row, l_original - Step)
						l_numeric := (l_numeric - l_network.loss (l_input, l_target)) / (2.0 * Step)
						al_dense.set_bias (l_row, l_original)
						l_max_error := l_max_error.max (relative_error (l_numeric, al_dense.bias_gradient (l_row)))
						l_checked := l_checked + 1
						l_row := l_row + 1
					end
				end
				l_layer_index := l_layer_index + 1
			end

			print ("Gradient Check (3-4-2, tanh/sigmoid, central differences, step " + Step.out + "):%N")
			print ("  Parameters checked: " + l_checked.out + "%N")
			print ("  Max relative error: " + l_max_error.out + " (must be < " + Tolerance.out + ")%N")
			assert ("all 26 parameters checked", l_checked = 26)
			assert ("backprop matches finite differences", l_max_error < Tolerance)
		end

	test_batch_gradients_accumulate
			-- Two backward passes without an update sum their gradients.
		local
			l_network: NEURAL_NETWORK
			l_single: REAL_64
		do
			create l_network.make
			l_network.add_layer (create {DENSE_LAYER}.make_seeded (2, 1, 5))
			l_network.add_layer (create {ACTIVATION_LAYER}.make_sigmoid (1))
			l_network.compute_gradients (<<1.0, 0.5>>, <<1.0>>)
			if attached {DENSE_LAYER} l_network.get_layer (1) as al_dense then
				l_single := al_dense.weight_gradient (1, 1)
				assert ("non-zero gradient", l_single /= 0.0)
					-- Second pass on top of the first, without clearing: output gradient 1.0
					-- at the cached input <<1.0, 0.5>> adds 1.0 * 1.0 to weight (1, 1).
				al_dense.backward (<<1.0>>).do_nothing
				assert ("two samples accumulated", al_dense.accumulated_samples = 2)
				assert ("gradient summed", (al_dense.weight_gradient (1, 1) - (l_single + 1.0 * 1.0)).abs < 1.0e-12)
				al_dense.clear_gradients
				assert ("cleared", al_dense.accumulated_samples = 0 and al_dense.weight_gradient (1, 1) = 0.0)
			else
				assert ("layer 1 is dense", False)
			end
		end

feature {NONE} -- Implementation

	Step: REAL_64 = 1.0e-5
			-- Finite-difference step.

	Tolerance: REAL_64 = 1.0e-6
			-- Largest acceptable relative error.

	relative_error (a_numeric, a_analytic: REAL_64): REAL_64
			-- |a_numeric - a_analytic| relative to their size (absolute below 1e-4).
		do
			Result := (a_numeric - a_analytic).abs / (a_numeric.abs + a_analytic.abs).max (1.0e-4)
		ensure
			non_negative: Result >= 0.0
		end

end
