# reachbot_rl

This repository contains the code for training a reinforcement learning model to control the ReachBot Robot developed by the BDSML and ASL chair at Stanford. The model is trained using the MuJoCo physics engine. More specifically we use mujoco playground and insert our own robot model to train the model.

## Installation

1. Create a virtual environment with Python. Tested with Python 3.12 but other versions should work as well.

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

## Training

### Local

In the terminal, run the following command to train the model.

1. Uncomment the following line in train.py if you plan on using cpu rather than an nvidia gpu.
   ```python
   # os.environ['JAX_PLATFORM_NAME'] = 'cpu'
   ```
2. Run the following command to train the model.
   ```bash
   python train.py
   ```

### Google Colab

- Clone repo into your google drive.
- Open the `train_reachbot_in_colab.ipynb` notebook in Google Colab.
- Change runtime to GPU.
- Run all install cells and when asked to mount google drive, click on the link and authenticate.
- Run the remaining cells in the notebook to train the model.
