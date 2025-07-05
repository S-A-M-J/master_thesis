# Reward Variance Reduction for Cave Exploration Task

## Problem Analysis

From the training logs, we observed:
- `distance_to_target` rewards improving from 17.37 to 1315.41, but with very high std (40.78 to 1124.69)
- `vel_to_target` rewards improving from -1.50 to 59.63, but with very high std (1.88 to 84.05)
- Standard deviations not decreasing despite mean rewards improving

## Root Causes

1. **Exponential reward functions**: The original `_reward_dist_to_target` and `_reward_vel_to_target` used `exp(distance_diff) - 1.0` which creates extreme values
2. **Unbounded rewards**: No clipping or normalization of reward values
3. **Scale mismatch**: Reward scales (10.0 and 100.0) were too high for the actual reward ranges
4. **Environment diversity**: Different cave configurations may cause inconsistent performance

## Solutions Implemented

### 1. Reward Function Reshaping

**Before:**
```python
# Exponential reward causing high variance
return jp.exp(distance_diff) - 1.0

# Combined exponential + linear with no bounds
exp_dist_reward = jp.clip(jp.exp(dist_diff) - 1.0, min=0.0, max=None)
linear_dist_reward = dist_diff
return exp_dist_reward + linear_dist_reward
```

**After:**
```python
# Bounded reward using tanh (configurable)
normalized_diff = distance_diff / self._config.reward_config.max_step_distance
return jp.tanh(normalized_diff * self._config.reward_config.vel_reward_scale)

# Clipped linear + small proximity bonus
clipped_reward = jp.clip(dist_diff, -1.0, 1.0)
proximity_bonus = 0.1 * (1.0 / normalized_current_dist - 1.0)
return clipped_reward + proximity_bonus
```

### 2. Configuration Parameters Added

```python
reward_config = config_dict.create(
    use_shaped_rewards=True,      # Enable bounded rewards
    max_step_distance=0.2,        # Expected max step for normalization
    max_cave_distance=50.0,       # Max expected cave distance
    vel_reward_scale=2.0,         # Tanh scaling factor
    scales=config_dict.create(
        distance_to_target=5.0,   # Reduced from 10.0
        vel_to_target=10.0,       # Reduced from 100.0
        # ... other scales unchanged
    )
)
```

### 3. Reduced Reward Scales

- `distance_to_target`: 10.0 → 5.0
- `vel_to_target`: 100.0 → 10.0

Since rewards are now bounded approximately between -1 and 1, these scales are more appropriate.

## Expected Results

1. **Lower standard deviations**: Bounded rewards should significantly reduce variance
2. **More stable learning**: Consistent reward ranges should improve policy gradient estimates
3. **Better convergence**: Reduced noise in reward signal should lead to more stable learning
4. **Maintained performance**: The shaped rewards still encourage the same behavior

## Alternative Approaches to Consider

If variance remains high, consider:

1. **Reward normalization**: Track running mean/std of rewards and normalize
2. **Curriculum learning**: Start with easier targets, gradually increase difficulty
3. **Multiple targets**: Use intermediate waypoints instead of single distant target
4. **Exploration bonuses**: Add structured exploration rewards to encourage consistent behavior
5. **Value function clipping**: Clip value estimates in the RL algorithm
6. **Different algorithms**: Try algorithms like PPO with entropy regularization or SAC

## Monitoring

To evaluate effectiveness:
1. Monitor std values in training logs - they should decrease over time
2. Plot reward distributions - should become less skewed
3. Check episode length variance - should also stabilize
4. Evaluate policy consistency across different cave configurations

## Usage

The changes are backward compatible. To use the original high-variance rewards:
```python
config_overrides = {"reward_config.use_shaped_rewards": False}
```

To fine-tune the new shaped rewards:
```python
config_overrides = {
    "reward_config.max_step_distance": 0.15,  # Adjust based on robot capabilities
    "reward_config.vel_reward_scale": 3.0,    # Increase for more sensitivity
    "reward_config.max_cave_distance": 40.0   # Adjust based on cave size
}
```
