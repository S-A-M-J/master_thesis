
# How to run Cave Exploration RL Training

1. Run `generation/create_cave_sets.ipynb` to generate cave environments and save them. You can also vary the cave parameters in the notebook to create different cave sets.

2. Run `training/run_cave_exploration.py` to train a reinforcement learning agent on the cave exploration task. The script includes all necessary steps for environment setup, training loop, logging, checkpointing, and video rendering.

The following parameters can be given as command line arguments to `training/run_cave_exploration.py`:
- `--caves_dir`: Path to the cave set file generated in step 1.
- `--eval_cave_id`: ID of the cave to use for evaluation.

3. Run `training/continue_training.py` to continue training from a saved checkpoint. Provide the path to the checkpoint file using the `--log_dir` argument.

The following parameters can be given as command line arguments to `training/continue_training.py`:
- `--log_dir`: Path to the log directory containing the checkpoint to continue from. The latest checkpoint in the directory will be used.
- `--additional_timesteps`: Additional timesteps to train (default: 50M).
- `--caves_directory`: Directory containing the cave environments from step 1 that you want to train on.

# How to run evaluation
Run `evaluation/evaluation_run.py` to evaluate a trained agent on a set of cave environments. 

The following parameters can be given as command line arguments to `evaluation/evaluation_run.py`:
- `--log_dir`: Path to the training run folder (must contain `config.json`, `params/`, and `checkpoints/`).
- `--checkpoint`: Optional path to a specific checkpoint directory to load parameters from.
- `--caves_directory`: Override caves directory (if not provided, uses the value from `config.json`).
- `--episodes_per_cave`: Number of episodes to run per cave (default: 3).
- `--max_steps`: Maximum steps per episode (default: 4000).
- `--seed`: Base random seed (default: 0).

# Tips
 
 - Use `tmux` or `screen` to run training in the background and do not forget to redirect the output to a log file. Example command:
   ```bash
   nohup python -u training/run_cave_exploration.py --caves_dir environment/generation/caves --eval_cave_id 280 > cave_training.log 2>&1 &
   ```
- Use eval_cave_id > 270 as these caves are not included in the training set.

# Cave Exploration Task Directory Structure
This directory contains the implementation and assets for the ReachBot robot's cave exploration task. Below is a description of the main files and folders:

- **cave_exploration.py**: Main environment implementation for the cave exploration task, including simulation setup and logic.

- **environment/**: Contains environment setup, domain randomization, scene files, and utilities.
	- `scene_reachbot_cave.xml`: MuJoCo XML scene file for cave exploration.
	- `domain_randomize.py`, `env_loader.py`: Scripts for domain randomization and environment loading.
	- **generation/**: Cave generation scripts and assets.
		- `cave_generator.py`: Script for generating cave environments.
		- `create_cave_sets.ipynb`: Notebook for creating cave sets.
		- **caves/**: Generated cave environment data, organized by ID.
	- **utils/**: Utility scripts and notebooks for cave analysis and rendering.
		- `analyze_cave.ipynb`, `display_home_keyframe.py`, `render_cave.py`: Tools for analyzing and visualizing caves.

- **evaluation/**: Contains evaluation scripts.
	- `evaluation_run.py`: Script to evaluate trained agents on cave environments.

- **tools/**: Analysis and rendering tools for episodes and policies.
	- `analyze_episode.ipynb`, `analyze_lidar.py`, `render_policy.py`: Notebooks and scripts for episode and policy analysis.

- **training/**: Training scripts for RL agents.
	- `run_cave_exploration.py`: Main training script for cave exploration RL.
	- `continue_training.py`: Script to continue training from a checkpoint.

This structure supports simulation, training, evaluation, and analysis of the cave exploration task for the ReachBot robot, including environment generation, assets, and training scripts.
