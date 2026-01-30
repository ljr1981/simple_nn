note
	description: "Test application for simple_nn neural network library"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class TEST_APP

create
	make

feature {NONE} -- Initialization

	make
			-- Run all tests.
		do
			print ("Running simple_nn tests...%N%N")
			passed := 0
			failed := 0

			run_xor_test

			print ("%N========================%N")
			print ("Results: " + passed.out + " passed, " + failed.out + " failed%N")

			if failed > 0 then
				print ("TESTS FAILED%N")
			else
				print ("ALL TESTS PASSED%N")
			end
		end

feature {NONE} -- Test Runners

	run_xor_test
		local
			l_tests: TEST_XOR_PROBLEM
		do
			print ("XOR Problem Tests:%N")
			create l_tests
			run_test (agent l_tests.test_xor_learning, "test_xor_learning")
			print ("%N")
		end

feature {NONE} -- Test Execution

	run_test (a_test: PROCEDURE; a_name: STRING)
			-- Execute a single test.
		do
			a_test.call ([])
			print ("  PASS: " + a_name + "%N")
			passed := passed + 1
		rescue
			print ("  FAIL: " + a_name + " (exception)%N")
			failed := failed + 1
		end

feature {NONE} -- State

	passed: INTEGER
			-- Number of passed tests.

	failed: INTEGER
			-- Number of failed tests.

end
