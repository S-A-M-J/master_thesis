# reachbot_rl

This repository contains the code for training a reinforcement learning model to control the ReachBot Robot developed by the BDSML and ASL chair at Stanford. The model is trained using the MuJoCo physics engine. More specifically we use mujoco playground and insert our own robot model to train the model.

## Installation

0. Requirements:

   Make sure to have cuda 12 or higher installed wherever you're running the training.

1. Create a virtual environment with Python. Tested with Python 3.12 but other versions should work as well. Also would recommend using conda but venv should work fine.

   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment.

   ```bash
   # On macOS/Linux
   source venv/bin/activate

   # On Windows
   .\venv\Scripts\activate
   ```

3. Clone the repository.

   ```bash
   git clone https://github.com/your-username/reachbot_rl.git
   cd reachbot_rl
   ```

4. Install requirements with `pip install -r requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

5. In VS Code change the .vscode/settings.json to use mjpython instead of python3.12.
   ```json
   {
     "python.defaultInterpreterPath": "<your_path>/<virtual_env>/bin/mjpython"
   }
   ```

# How to use
For specific usage instructions, please refer to the respective task directory README files.