note
	description: "[
		Shared helpers for simple_nn tests: an `assert' that raises a real
		exception (it does not depend on `check' monitoring, which a finalized
		test executable may not do) and the XOR data set.
	]"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class NN_TEST_SUPPORT

feature {NONE} -- Assertions

	assert (a_tag: STRING; a_condition: BOOLEAN)
			-- Fail the running test, reporting `a_tag', unless `a_condition' holds.
		require
			tag_not_empty: not a_tag.is_empty
		local
			l_failure: DEVELOPER_EXCEPTION
		do
			if not a_condition then
				print ("    ASSERTION FAILED: " + a_tag + "%N")
				create l_failure
				l_failure.set_description ("Test assertion failed: " + a_tag)
				l_failure.raise
			end
		ensure
			held: a_condition
		end

feature {NONE} -- XOR data

	xor_inputs: ARRAY [ARRAY [REAL_64]]
			-- The four XOR input pairs.
		do
			create Result.make_filled (create {ARRAY [REAL_64]}.make_empty, 1, 4)
			Result [1] := <<0.0, 0.0>>
			Result [2] := <<0.0, 1.0>>
			Result [3] := <<1.0, 0.0>>
			Result [4] := <<1.0, 1.0>>
		ensure
			four_samples: Result.count = 4
		end

	xor_targets: ARRAY [ARRAY [REAL_64]]
			-- XOR of each of `xor_inputs'.
		do
			create Result.make_filled (create {ARRAY [REAL_64]}.make_empty, 1, 4)
			Result [1] := <<0.0>>
			Result [2] := <<1.0>>
			Result [3] := <<1.0>>
			Result [4] := <<0.0>>
		ensure
			four_samples: Result.count = 4
		end

	format_real (a_value: REAL_64): STRING
			-- `a_value' rounded to 4 decimal places.
		do
			Result := ((a_value * 10000.0).rounded / 10000.0).out
		end

end
