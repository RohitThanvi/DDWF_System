Trained model checkpoints go here. This directory is intentionally tracked
in git (via this file + .gitkeep) so it exists after a fresh clone --
train_downscaler.py and train_global_engine.py both save directly into it
and will fail with "Parent directory checkpoints does not exist" if it's
missing.

The actual .pt/.pth/.ckpt weight files are gitignored (see .gitignore) --
they're large binary artifacts that don't belong in version control. Train
your own (see docs/TRAINING.md) or copy a checkpoint here manually.
