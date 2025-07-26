
# How to Run Cave Exploration RL Training
To run the Cave Exploration RL training script, follow these steps:
1. **Navigate to the Task Directory**:
   Open a terminal and change to the directory where the `run_cave_exploration.py` script is located.
   ```bash
   cd tasks/cave_exploration
   ``` 
2. **Run the Script**:
   Execute the script using Python. This will start the training process and log output to `cave_training.log`.
   ```bash
   nohup python -u run_cave_exploration.py > cave_training.log 2>&1 &
   ```
   - The `nohup` command allows the script to run in the background even if the terminal is closed.
   - The `-u` flag forces the stdout and stderr streams to be unbuffered, which is useful for real-time logging.
   - The output will be redirected to `cave_training.log`, allowing you to monitor the training progress later.
