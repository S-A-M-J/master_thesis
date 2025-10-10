#!/usr/bin/env python3
"""
Cave Exploration RL Training Script

This script contains the cave exploration reinforcement learning training pipeline,
converted from the Jupyter notebook for better modularity and production use.
"""

import random
import sys
import os
import threading
from datetime import datetime
import functools
import argparse

# Override print to always flush
print = functools.partial(print, flush=True)

# Add execution tracking to debug duplicate output
print("=== STARTING CAVE EXPLORATION RL TRAINING ===")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# GPU configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Configure JAX GPU memory settings BEFORE importing jax - OPTIMIZED FOR 40GB A100
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.88'  # Use 90% of GPU memory (~36GB out of 40GB)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate - grow as needed to avoid fragmentation

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import jax
from jax import numpy as jp
from jax.lib import xla_bridge

print("Device count: ", jax.device_count())
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"JAX memory fraction set to: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}")

print("JAX backend info:")
print(f"Platform: {xla_bridge.get_backend().platform}")
print(f"Device count: {xla_bridge.get_backend().device_count()}")
print(f"Devices: {xla_bridge.get_backend().devices()}")

# JAX configuration optimized for large workloads
jax.config.update('jax_enable_x64', False)  # Use float32 to save memory
jax.config.update('jax_traceback_filtering', 'off')

# Check GPU availability and memory
gpu_available = jax.devices()[0].platform == 'gpu'
print(f"GPU available: {gpu_available}")

if gpu_available:
    gpu_device = jax.devices('gpu')[0]
    print(f"GPU device: {gpu_device}")
else:
    print("No GPU device found.")

print("Basic imports completed...")
import mujoco
import json
import imageio
import gc
from pathlib import Path

print("Basic imports completed...")

# Brax and training imports
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from mujoco_playground.config import locomotion_params
from mujoco_playground import wrapper
from tensorboardX import SummaryWriter

print("Brax imports completed...")

# Task-specific imports
from tasks.cave_exploration.cave_exploration import CaveExplore, default_config as reachbot_config
from tasks.cave_exploration.environment.env_loader import CaveBatchLoader
from tasks.cave_exploration.environment.domain_randomize import create_cave_domain_randomizer, prepare_cave_data_arrays
from models.model_loader import ReachbotModelType
from tasks.common.randomize import domain_randomize as reachbot_randomize
from utils.telegram_messenger import send_message_sync
from utils.utils import JaxArrayEncoder, convert_to_dict

print("Task-specific imports completed...")

# Global variables
ENV_STR = 'Go1JoystickFlatTerrain'
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

print("=== ALL IMPORTS LOADED SUCCESSFULLY! ===")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cave Exploration RL Training Script')
    
    parser.add_argument('--caves_dir', 
                        type=str, 
                        default=os.path.join(os.path.dirname(__file__), "environment", "caves"),
                        help='Directory containing cave environments (default: environment/caves)')
    
    parser.add_argument('--eval_cave_id', 
                        type=int, 
                        default=None,
                        help='Specific cave ID to use for evaluation. If None, randomly selects from eval caves (default: None)')
    
    return parser.parse_args()

def configure_environment():
    """Configure the environment settings"""
    print("=== CONFIGURING ENVIRONMENT ===")
    
    env_cfg = reachbot_config()

    # Cave batch size is no longer needed - scenes are auto-split 90%/10%
    
    # Basic simulation parameters
    env_cfg.sim_dt = 0.004
    env_cfg.action_scale = 1
    env_cfg.stickiness_config.enable = True  # Enable stickiness forces
    
    # PID control parameters
    env_cfg.Kp_pri = 60.0
    env_cfg.Kd_pri = 20.0
    env_cfg.Kp_rot = 25.0
    env_cfg.Kd_rot = 2.0
    
    env_cfg.noise_config.level = 0.0

    env_cfg.reward_config.scales.track_lidar_direction = 1.0
    env_cfg.reward_config.scales.wide_stance = 0.0
    
    # Reward scaling configuration
    env_cfg.reward_config.scales.orientation = -1.0
    env_cfg.reward_config.scales.torques = 0.0
    env_cfg.reward_config.scales.action_rate = 0.0
    env_cfg.reward_config.scales.dof_pos_limits = 0.0
    env_cfg.reward_config.scales.energy = 0.0
    env_cfg.reward_config.scales.termination = -1.0
    env_cfg.reward_config.scales.inactivity = -0.1
    
    # Target-based rewards
    env_cfg.reward_config.scales.distance_from_start = -1
    env_cfg.reward_config.scales.stability = -0.5
    env_cfg.reward_config.scales.exploration_rate = 0.0
    env_cfg.reward_config.scales.milestone_reward = 0.1  # Reward for reaching x-direction milestones

    env_cfg.randomize_starting_pos = False  # Randomize starting position within the cave

    
    print("Environment configuration completed!")
    return env_cfg

def configure_ppo_parameters():
    """Configure PPO training parameters"""
    print("=== CONFIGURING PPO PARAMETERS ===")
    
    ppo_params = locomotion_params.brax_ppo_config(ENV_STR)
    ppo_training_params = dict(ppo_params)
    # Modify params for training
    ppo_training_params["num_timesteps"] = 20_000_000  # 100 million timesteps
    ppo_training_params["episode_length"] = 4000
    ppo_training_params["num_envs"] = 2048
    ppo_training_params["batch_size"] = 256
    ppo_training_params["num_minibatches"] = 32
    ppo_training_params["num_updates_per_batch"] = 4
    ppo_training_params["unroll_length"] = 120
    ppo_training_params["entropy_cost"] = 0.02
    ppo_training_params["learning_rate"] = 3e-4
    ppo_training_params["discounting"] = 0.995
    #ppo_training_params["clipping_epsilon"] = 0.2
    ppo_training_params["num_evals"] = ppo_training_params["num_timesteps"] // 10_000_000
    if (ppo_training_params["num_evals"] < 10):
        ppo_training_params["num_evals"] = 10
    
    print("PPO training parameters:")
    for key, value in ppo_training_params.items():
        print(f"  {key}: {value}")
    
    print("PPO parameters configuration completed!")
    return ppo_params, ppo_training_params

def save_video(frames, video_path, fps):
    import imageio
    imageio.mimsave(video_path, frames, fps=fps)
       

def trainModel(ppo_params_input:dict, env_cfg, cave_batch_loader, caves_directory):
    """Main training function that matches the notebook approach"""
    
    # Create log directory for training run
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join(os.getcwd(), "logs/cave_exploration-"+datetime_str)
    os.makedirs(logdir, exist_ok=True)
    
    # Load cave environments using CaveBatchLoader
    print("Loading cave environments...")
    
    
    # Print dataset summary
    dataset_summary = cave_batch_loader.get_dataset_summary()
    print(f"\nDataset Summary:")
    print(f"  Total caves: {dataset_summary['total_caves']}")
    print(f"  Training caves: {dataset_summary['training_caves']['count']}")
    print(f"  Evaluation caves: {dataset_summary['eval_caves']['count']}")
    print(f"  No overlap: {dataset_summary['no_overlap']}")
    print(f"  Training master cave: {dataset_summary['training_master_cave']}")
    print(f"  Evaluation master cave: {cave_batch_loader.eval_scene['master_cave_id']}")


    training_scene_data = cave_batch_loader.get_training_scene_data()
    eval_scene_data = cave_batch_loader.get_eval_scene_data()

    # Create training environment with domain randomization enabled
    env = CaveExplore(
        config=env_cfg, 
        scene_type="training",
        scene_data=training_scene_data, 
        domain_randomization_enabled=True,
    )

    # For evaluation, use a fixed cave without domain randomization
    selected_eval_cave_id = cave_batch_loader.get_master_cave_id(scene_type="eval")
    eval_env = CaveExplore(
        config=env_cfg,
        scene_type="evaluation",
        scene_data=eval_scene_data,
    )
    eval_env.select_cave_environment(selected_eval_cave_id)
    
    print(f"\nEnvironment creation completed:")
    print(f"  Training environment: Domain randomization with {len(cave_batch_loader.training_scene['caves'])} caves")
    print(f"  Evaluation environment: Cave {selected_eval_cave_id}")

    # Setup domain randomization for cave environments
    print("\n=== SETTING UP CAVE DOMAIN RANDOMIZATION ===")
    try:
        print("Preparing cave data arrays...")
        cave_data_arrays = prepare_cave_data_arrays(cave_batch_loader, max_boxes=7500)
        
        print("Creating domain randomizer...")
        cave_domain_randomize = create_cave_domain_randomizer(cave_data_arrays, max_boxes=7500)

        print(f"Domain randomization setup completed:")
        print(f"  Total caves for randomization: {cave_data_arrays['all_cave_positions'].shape[0]}")
        print(f"  Max boxes per cave: {cave_data_arrays['all_cave_positions'].shape[1]}")
        print(f"  Cave wall geom IDs: {len(cave_data_arrays['cave_wall_geom_ids'])}")
        
        print("Cave domain randomization function created successfully!")
        
    except Exception as e:
        print(f"Warning: Failed to setup domain randomization: {e}")
        print("Falling back to standard domain randomization")
        import traceback
        traceback.print_exc()
        cave_domain_randomize = reachbot_randomize

    # Initialize tracking variables
    timesteps = []
    rewards = []
    total_rewards = []
    total_rewards_std = []
    times = [datetime.now()]
    
    writer = SummaryWriter(logdir=logdir)
    
    # Save configurations
    print("Saving configs")
    
    configs = {
        "env_cfg": convert_to_dict(env_cfg),
        "ppo_params": convert_to_dict(ppo_params_input),
        "dataset_summary": dataset_summary,
        "selected_eval_cave_id": selected_eval_cave_id,
        "training_caves_ids": list(cave_batch_loader.training_scene["caves"].keys()),
        "eval_cave_ids": list(cave_batch_loader.eval_scene["caves"].keys()),
        "reachbot_model_type": ReachbotModelType.BASIC.name,
        "training_mode": "wrapper_dynamic_caves",
        "domain_randomization": {
            "enabled": 'cave_data_arrays' in locals(),
            "num_caves": cave_data_arrays['all_cave_positions'].shape[0] if 'cave_data_arrays' in locals() else 0,
            "max_boxes_per_cave": cave_data_arrays['all_cave_positions'].shape[1] if 'cave_data_arrays' in locals() else 0,
            "cave_wall_geom_ids_count": len(cave_data_arrays['cave_wall_geom_ids']) if 'cave_data_arrays' in locals() else 0,
        },
        "caves_directory": caves_directory
    }
    
    config_path = os.path.join(logdir, 'config.json')
    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(configs, fp, indent=4, cls=JaxArrayEncoder)
    print(f"Configuration saved to {config_path}")
    writer.add_text('config', json.dumps(configs, indent=4))
    
    # Progress tracking function
    def progress(num_steps, metrics):
        """Function to track progress and log metrics during training."""
        print(f"Progress at step {num_steps}: {metrics}")
        # Log to TensorBoard
        for key, value in metrics.items():
            if not (jp.isnan(value) or jp.isinf(value)):
                writer.add_scalar(key, value, num_steps)
            else:
                print(f"Warning: Skipping NaN/Inf value for metric '{key}' at step {num_steps}")
        
        if "eval/episode_reward" in metrics:
            episode_reward = metrics["eval/episode_reward"]
            if jp.isnan(episode_reward) or jp.isinf(episode_reward):
                print("Warning: NaN/Inf reward encountered, aborting.")
                run_duration = str(datetime.now() - times[0])
                send_message_sync(
                    task="Cave Exploration RL Training",
                    duration=run_duration,
                    result="Failed: NaN/Inf reward encountered"
                )
                raise ValueError(f"NaN/Inf reward encountered at step {num_steps}: {episode_reward}")
            
            times.append(datetime.now())
            timesteps.append(num_steps)
            total_rewards.append(episode_reward)
            total_rewards_std.append(metrics["eval/episode_reward_std"])
        
            writer.flush()
            metrics["timesteps"] = num_steps
            metrics["time"] = (times[-1] - times[0]).total_seconds()
            rewards.append(metrics)
            
            percent_complete = (num_steps / ppo_params_input["num_timesteps"]) * 100
            if num_steps == 0:
                remaining_time_str = "unknown"
            else:
                remaining_time = (ppo_params_input["num_timesteps"] - num_steps) * (times[-1] - times[0]).total_seconds() / num_steps / 60
                remaining_time_str = f"{remaining_time:.2f}"
            
            print(f"step: {num_steps}/{ppo_params_input['num_timesteps']} ({percent_complete:.1f}%), reward: {total_rewards[-1]:.3f} +/- {total_rewards_std[-1]:.3f}, time passed (min): {(times[-1] - times[0]).total_seconds() / 60:.2f} min, calculated time left (min): {remaining_time_str} min")
    
    # Network factory setup
    print("Input layer size:", env.observation_size)
    print("Output layer size:", env.action_size)
    network_factory = ppo_networks.make_ppo_networks(observation_size=env.observation_size, action_size=env.action_size)
    
    ppo_params = locomotion_params.brax_ppo_config(ENV_STR)
    if "network_factory" in ppo_params:
        if "network_factory" in ppo_params_input:
            del ppo_params_input["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )
    

    
    # Wrap the progress function to include cave logging
    def policy_params_fn(current_step, make_policy, params):
        del make_policy  # Unused.
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        checkpoint_path = os.path.join(logdir, 'checkpoints')
        path = os.path.join(checkpoint_path, f"{current_step}")
        abs_path = os.path.abspath(path)
        orbax_checkpointer.save(abs_path, params, force=True, save_args=save_args)
    
    
    train_fn = functools.partial(
        ppo.train, 
        **dict(ppo_params_input),
        network_factory=network_factory,
        progress_fn=progress,  # Use wrapped progress function
        policy_params_fn=policy_params_fn,
        randomization_fn=cave_domain_randomize,  # Add cave domain randomization
        max_devices_per_host=1,
        log_training_metrics=True,
    )
    
    # Run training
    print("Training the model...")
    try:
        make_inference_fn, params, metrics = train_fn(
            environment=env,
            eval_env=eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        print("Training completed successfully!")
    except Exception as e:
        import traceback
        run_duration = str(datetime.now() - times[0])
        send_message_sync(
            task="Cave Exploration RL Training",
            duration=run_duration,
            result=f"Failed: {e}"
        )
        traceback.print_exc()
        raise
    
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    
    # Save results
    results_path = os.path.join(logdir, 'results.txt')
    with open(results_path, 'w') as f:
        for i in range(len(total_rewards)):
            f.write(f"step: {timesteps[i]}, reward: {total_rewards[i]}, reward_std: {total_rewards_std[i]}\n")
        f.write(f"Time to jit: {times[1] - times[0]}\n")
        f.write(f"Time to train: {times[-1] - times[1]}\n")
    
    # Save rewards as JSON
    def nest_flat_dict(flat_dict):
        nested_dict = {}
        for key, value in flat_dict.items():
            parts = key.split('/')
            d = nested_dict
            for i, part in enumerate(parts):
                is_last_part = (i == len(parts) - 1)
                if is_last_part:
                    if isinstance(d.get(part), dict):
                        d[part]['value'] = value
                    else:
                        d[part] = value
                else:
                    if not isinstance(d.get(part), dict):
                        d[part] = {'value': d[part]} if part in d else {}
                    d = d[part]
        return nested_dict
    
    nested_rewards = [nest_flat_dict(r) for r in rewards]
    rewards_path = os.path.join(logdir, 'rewards.json')
    with open(rewards_path, 'w') as fp:
        json.dump(nested_rewards, fp, indent=4, cls=JaxArrayEncoder)
    
    # Save trained parameters
    params_path = os.path.join(logdir, 'params')
    model.save_params(params_path, params)
    
    print(f"Training completed! Results saved to: {logdir}")
    print(f"Final reward: {total_rewards[-1]:.3f} ± {total_rewards_std[-1]:.3f}")
    
    # Store these variables for the video generation
    trained_params = params
    trained_make_inference_fn = make_inference_fn
    trained_env = eval_env
    trained_logdir = logdir
    
    # Free up training memory before rendering
    del train_fn, network_factory, writer
    gc.collect()
    
    return trained_env, trained_params, trained_make_inference_fn, trained_logdir, times, total_rewards, total_rewards_std



def create_videos(env, params, make_inference_fn, logdir, times, total_rewards, total_rewards_std):
    """Create videos from the trained model"""
    print("=== CREATING VIDEOS ===")
    
    # Free up training memory before rendering
    gc.collect()
    
    # Setup JIT compiled functions for inference
    # Note: This should work fine since we're using the evaluation environment (not training env)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    
    print("Setting up rollout for video creation...")
    
    # Rollout parameters
    rng = jax.random.PRNGKey(0)
    n_episodes = 5
    rollout_steps = 2000
    
    # Rollout policy and record simulation
    print(f"Running rollout for {n_episodes} episode(s) with {rollout_steps} steps each...")
    episode_rewards = []
    
    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}")
        episode_rng, rng = jax.random.split(rng)
        # Use JIT compiled reset function (should work fine with eval environment)
        state = jit_reset(episode_rng)
        rollout = [state]  # Reset rollout for each episode
        episode_reward = 0.0
        
        for i in range(rollout_steps):
            if i % 500 == 0:
                print(f"  Step {i}/{rollout_steps}, Current reward: {episode_reward:.3f}")
                
            act_rng, rng = jax.random.split(episode_rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            
            # Check for numerical issues
            if jp.any(jp.isinf(ctrl)) or jp.any(jp.isnan(ctrl)):
                print(f"Numerical issue detected in control at step {i}. Stopping rollout.")
                break
                
            state = jit_step(state, ctrl)
            
            # Accumulate reward for this episode
            episode_reward += float(state.reward)
            
            if state.done:
                print(f"Episode {episode + 1} ended at step {i} with reward: {episode_reward:.3f}")
                break
                
            rollout.append(state)
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1} completed with {len(rollout)} states and total reward: {episode_reward:.3f}")
        
        # Render video
        print("Rendering video...")
        render_every = 1  # Render every frame
        width = 1920      # Full HD width
        height = 1080     # Full HD height
        
        frames = env.render(rollout[::render_every], camera='track_global', width=width, height=height)
        print(f"Rendered {len(frames)} frames")
        
        # Save video
        video_path = os.path.join(logdir, f'posttraining_episode_{episode}_reward_{episode_reward:.1f}.mp4')
        fps = 1.0 / env.dt
        
        print(f"Saving video to {video_path} at {fps} FPS...")
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Video saved successfully: Episode {episode + 1}, Reward: {episode_reward:.3f}")
    
    # Print summary of all episodes
    print("\n=== EPISODE REWARD SUMMARY ===")
    for i, reward in enumerate(episode_rewards):
        print(f"Episode {i + 1}: {reward:.3f}")
    print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.3f}")
    print(f"Best episode: {episode_rewards.index(max(episode_rewards)) + 1} with reward {max(episode_rewards):.3f}")
    print(f"Worst episode: {episode_rewards.index(min(episode_rewards)) + 1} with reward {min(episode_rewards):.3f}")
    
    # Send completion notification
    run_duration = str(times[-1] - times[0])
    if total_rewards:
        training_result = f"Training final reward: {total_rewards[-1]:.3f} ± {total_rewards_std[-1]:.3f}"
    else:
        training_result = "No training rewards recorded."
    
    # Include episode rewards in notification
    episode_summary = f"Episode rewards: {[f'{r:.1f}' for r in episode_rewards]}, Avg: {sum(episode_rewards)/len(episode_rewards):.1f}"
    
    send_message_sync(
        task="Cave Exploration RL Training",
        duration=run_duration,
        result=f"{training_result}\n{episode_summary}"
    )
    
    print("\n=== TRAINING AND VIDEO CREATION COMPLETE ===")
    print(f"Log directory: {logdir}")
    print(f"Training duration: {run_duration}")
    print(f"Training result: {training_result}")
    print(f"Episode summary: {episode_summary}")
    
    return episode_rewards

def main():
    """Main function to run the complete training pipeline"""
    print("🚀 Starting Cave Exploration RL Training Script")
    print(f"📅 Start time: {datetime.now()}")
    print(f"🔥 PID: {os.getpid()} | Thread: {threading.current_thread().ident}")
    
    # Parse command line arguments
    args = parse_arguments()
    print(f"📂 Caves directory: {args.caves_dir}")
    print(f"🔍 Evaluation cave ID: {args.eval_cave_id if args.eval_cave_id is not None else 'Random selection'}")
    
    try:
        # Configuration
        env_cfg = configure_environment()
        ppo_params, ppo_training_params = configure_ppo_parameters()

        cave_batch_loader = CaveBatchLoader(env_cfg, 
                                           reachbot_model_type=ReachbotModelType.BASIC, 
                                           caves_directory=args.caves_dir,
                                           eval_cave_index=args.eval_cave_id)
        
        # Call the updated training function 
        _, params, make_inference_fn, logdir, times, total_rewards, total_rewards_std = trainModel(ppo_training_params, env_cfg, cave_batch_loader, args.caves_dir)

        # Create a fresh, clean evaluation environment for video creation (like in render_policy.py)
        # This avoids tracer leaks from the training-contaminated eval_env
        print("Creating fresh evaluation environment for video rendering...")
        eval_scene_data = cave_batch_loader.get_eval_scene_data()
        selected_eval_cave_id = cave_batch_loader.get_master_cave_id(scene_type="eval")
        fresh_eval_env = CaveExplore(
            config=env_cfg,
            scene_type="evaluation", 
            scene_data=eval_scene_data
        )

        fresh_eval_env.select_cave_environment(selected_eval_cave_id)
        print(f"Fresh eval environment created for cave: {selected_eval_cave_id}")

        # Video creation using fresh evaluation environment (clean, no training contamination)
        episode_rewards = create_videos(fresh_eval_env, params, make_inference_fn, logdir, times, total_rewards, total_rewards_std)
        
        print("✅ All tasks completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise

# Main function
if __name__ == '__main__':
    main()