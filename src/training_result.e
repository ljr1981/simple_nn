note
	description: "Result of neural network training with loss history"
	author: "Larry Rix"
	date: "$Date$"
	revision: "$Revision$"

class TRAINING_RESULT

create
	make

feature {NONE} -- Initialization

	make
			-- Create empty training result.
		do
			create epochs.make (0)
			create losses.make (0)
		ensure
			empty: epochs.is_empty and losses.is_empty
		end

feature -- Access

	add_epoch (a_epoch: INTEGER; a_loss: REAL_64)
			-- Record loss for an epoch.
		require
			positive_epoch: a_epoch > 0
			non_negative_loss: a_loss >= 0.0
		do
			epochs.extend (a_epoch)
			losses.extend (a_loss)
		ensure
			added: epochs.count = old epochs.count + 1
			losses_count_matches: losses.count = epochs.count
		end

	epoch_count: INTEGER
			-- Number of training epochs.
		do
			Result := epochs.count
		end

	loss_at (a_epoch: INTEGER): REAL_64
			-- Loss value at specified epoch.
		require
			valid_epoch: a_epoch >= 1
		do
			Result := losses [a_epoch]
		end

	initial_loss: REAL_64
			-- Loss at first epoch.
		do
			Result := losses [1]
		end

	final_loss: REAL_64
			-- Loss at last epoch.
		do
			Result := losses [losses.count]
		end

feature {NONE} -- Implementation

	epochs: ARRAYED_LIST [INTEGER]
			-- Epoch numbers.

	losses: ARRAYED_LIST [REAL_64]
			-- Loss values corresponding to epochs.

invariant
	counts_match: epochs.count = losses.count

end
