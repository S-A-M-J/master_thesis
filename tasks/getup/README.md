
# How to train

Run the `train_getup_ppo.ipynb` Jupyter notebook to train a PPO agent on the getup task. The notebook includes all necessary steps for environment setup, training loop, logging, checkpointing, and video rendering.

# Getup Task Directory Structure

This directory contains the implementation and assets for the Go1 robot's fall recovery (getup) task. Below is a description of the main files and folders:

- **getup.py**: Implements the getup environment for the Go1 robot, including simulation setup, reward functions, and environment logic using JAX and MuJoCo.

- **environment/**: Contains simulation assets and XML scene files.
	- `getup_scene_template.xml`: Template MuJoCo XML scene file with placeholders for the robot model path.
	- `getup_scene.xml`: Generated MuJoCo XML scene file with the robot model path filled in.
	- **assets/**: Image assets for terrain and textures.
		- `hfield.png`: Height field texture for terrain.
		- `rocky_texture.png`: Rocky terrain texture.

- **train_getup_ppo.ipynb**: Jupyter notebook for training a PPO agent on the getup task. Includes environment setup, training loop, logging, checkpointing, and video rendering.


