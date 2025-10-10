
# How to train
Run the `train_joystick.py` Jupyter notebook to train a reinforcement learning agent on the joystick task. The notebook includes all necessary steps for environment setup, training loop, logging, checkpointing, and video rendering.

# Joystick Task Directory Structure

This directory contains the implementation and assets for the Go1 robot's joystick control task. Below is a description of the main files and folders:

- **joystick.py**: Implements the joystick environment for the Go1 robot, including simulation setup, reward functions, and environment logic using JAX and MuJoCo.

- **environment/**: Contains simulation assets and XML scene files.
	- `joystick_scene_template.xml`: Template MuJoCo XML scene file with placeholders for the robot model path.
	- `joystick_scene.xml`: Generated MuJoCo XML scene file with the robot model path filled in.
	- **assets/**: Image assets for terrain and textures.
		- `hfield.png`: Height field texture for terrain.
		- `rocky_texture.png`: Rocky terrain texture.

- **train_joystick.ipynb**: Jupyter notebook for training a reinforcement learning agent on the joystick task. Includes environment setup, training loop, logging, checkpointing, and video rendering.

- **train_joystick.py**: Python script for training the joystick task agent (alternative to the notebook).

- **README.md**: This documentation file.

This structure supports simulation, training, and evaluation of the joystick control task for the Go1 robot, including environment assets and training scripts.
