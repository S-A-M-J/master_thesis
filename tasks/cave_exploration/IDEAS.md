# Ideas for Cave Exploration Task

1. 
## Idea
Increase unroll length to 512 and reduce overall timesteps
## Why
Longer unrolls can capture more complex dependencies and interactions in the cave environment, potentially leading to better exploration strategies.
## Downside
Increased computational cost and memory usage, which may lead to longer training times and require more powerful hardware.

2.
## Idea
Fix max speed of robot to 0.5 m/s and normalize/clip rewards accordingly
## Why
Normalizing or clipping rewards can help stabilize training and prevent large reward spikes that can destabilize learning
## Downside
This may limit the learning speed of the robot, as it may not be able to explore the environment as quickly as it could with a higher speed.
