#!/usr/bin/env python3
"""
Cave Exploration RL Training Continuation Script

This script continues training from a previously saved checkpoint,
loading the configuration from the original training run.
"""

import random
import sys
import os
import threading
import argparse
from datetime import datetime
import functools

# Override print to always flush
print = functools.partial(print, flush=True)

# Add execution tracking to debug duplicate output
print("=== STARTING CAVE EXPLORATION RL TRAINING ===")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# GPU configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Configure JAX GPU memory settings BEFORE importing jax - OPTIMIZED FOR 40GB A100
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.985'  # Use 98.5% of GPU memory (~39.4GB out of 40GB)
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
from ml_collections import config_dict

print("Brax imports completed...")

# Task-specific imports
from tasks.cave_exploration.cave_exploration import CaveExplore, default_config as reachbot_config
from tasks.cave_exploration.environment.env_loader import CaveBatchLoader
from tasks.cave_exploration.domain_randomize import create_cave_domain_randomizer, prepare_cave_data_arrays
from models.model_loader import ReachbotModelType
from tasks.common.randomize import domain_randomize as reachbot_randomize
from utils.telegram_messenger import send_message_sync

print("Task-specific imports completed...")

# Global variables
ENV_STR = 'Go1JoystickFlatTerrain'
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

print("=== ALL IMPORTS LOADED SUCCESSFULLY! ===")

# JSON encoder for JAX arrays
class JaxArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, jp.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def convert_to_dict(obj):
    """Convert ConfigDict and other non-serializable objects to regular dicts"""
    if hasattr(obj, 'to_dict'):
        # Handle ConfigDict objects
        return convert_to_dict(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        # Handle objects with __dict__ attribute
        return convert_to_dict(obj.__dict__)
    elif isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_dict(v) for v in obj]
    elif isinstance(obj, jp.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and obj == float('inf'):
        return 1e308
    elif isinstance(obj, float) and obj == float('-inf'):
        return -1e308
    else:
        return obj

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Continue cave exploration RL training from checkpoint")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to the log directory containing the checkpoint to continue from")
    parser.add_argument("--additional_timesteps", type=int, default=50_000_000,
                        help="Additional timesteps to train (default: 50M)")
    return parser.parse_args()

def load_config_from_logdir(log_dir):
    """Load configuration from a log directory"""
    print(f"Loading configuration from: {log_dir}")
    
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        print(f"Error: Log directory does not exist: {log_dir}")
        print("Available training runs:")
        logs_base_dir = os.path.join(os.path.dirname(__file__), "../../logs")
        if os.path.exists(logs_base_dir):
            for run in sorted(os.listdir(logs_base_dir)):
                if run.startswith("cave_exploration-"):
                    print(f"  {run}")
        exit(1)
    
    # Load configuration from the checkpoint
    config_path = os.path.join(log_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {log_dir}")
        exit(1)
        
    with open(config_path, 'r') as f:
        loaded_config = json.load(f)
    
    print("Configuration loaded successfully!")
    return loaded_config

def find_latest_checkpoint(log_dir):
    """Find the latest checkpoint in the log directory"""
    checkpoints_dir = os.path.join(log_dir, 'checkpoints')
    
    if not os.path.exists(checkpoints_dir):
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        exit(1)
    
    # Get all checkpoint directories and sort by step number
    checkpoint_dirs = []
    for item in os.listdir(checkpoints_dir):
        item_path = os.path.join(checkpoints_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            checkpoint_dirs.append((int(item), item_path))
    
    if not checkpoint_dirs:
        print(f"Error: No checkpoint directories found in {checkpoints_dir}")
        exit(1)
    
    # Sort by step number and get the latest
    checkpoint_dirs.sort(key=lambda x: x[0])
    latest_step, latest_checkpoint_path = checkpoint_dirs[-1]
    
    print(f"Found {len(checkpoint_dirs)} checkpoints")
    print(f"Latest checkpoint: Step {latest_step} at {latest_checkpoint_path}")
    
    return latest_step, latest_checkpoint_path


def configure_environment_from_config(loaded_config):
    """Configure the environment settings from loaded config"""
    print("=== CONFIGURING ENVIRONMENT FROM LOADED CONFIG ===")
    
    # Get the default environment configuration and update with saved config
    env_cfg = reachbot_config()
    json_env_cfg = config_dict.ConfigDict(loaded_config['env_cfg'])
    env_cfg.update(json_env_cfg)
    
    print("Environment configuration loaded from config!")
    return env_cfg

def configure_ppo_parameters_from_config(loaded_config, additional_timesteps):
    """Configure PPO training parameters from loaded config"""
    print("=== CONFIGURING PPO PARAMETERS FROM LOADED CONFIG ===")
    
    # Load base PPO parameters from config
    ppo_params = locomotion_params.brax_ppo_config(ENV_STR)
    ppo_training_params = dict(loaded_config['ppo_params'])
    
    # Update with additional timesteps for continuation
    ppo_training_params["num_timesteps"] = additional_timesteps
    
    print("PPO training parameters for continuation:")
    for key, value in ppo_training_params.items():
        print(f"  {key}: {value}")
    
    print("PPO parameters configuration completed!")
    return ppo_params, ppo_training_params

def save_video(frames, video_path, fps):
    import imageio
    imageio.mimsave(video_path, frames, fps=fps)
       

def trainModel(ppo_params_input: dict, env_cfg, loaded_config, checkpoint_path=None, initial_step=0, original_log_dir=None):
    """Main training function that continues from checkpoint or starts fresh"""
    
    # Create log directory for training run
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if original_log_dir:
        # Extract original timestamp for reference
        original_timestamp = os.path.basename(original_log_dir).replace("cave_exploration-", "")
        logdir = os.path.join(os.getcwd(), f"logs/cave_exploration-continue-{original_timestamp}-{datetime_str}")
    else:
        logdir = os.path.join(os.getcwd(), "logs/cave_exploration-"+datetime_str)
    os.makedirs(logdir, exist_ok=True)
    
    print(f"New training log directory: {logdir}")
    if original_log_dir:
        print(f"Continuing from original run: {original_log_dir}")
        print(f"Starting from step: {initial_step}")
    
    # Load cave environments using CaveBatchLoader (same as before)
    print("Loading cave environments...")
    cave_batch_loader = CaveBatchLoader(env_cfg, ReachbotModelType.BASIC)
    
    # Print dataset summary
    dataset_summary = cave_batch_loader.get_dataset_summary()
    print(f"\nDataset Summary:")
    print(f"  Total caves: {dataset_summary['total_caves']}")
    print(f"  Training caves: {dataset_summary['training_caves']['count']}")
    print(f"  Evaluation caves: {dataset_summary['eval_caves']['count']}")
    print(f"  No overlap: {dataset_summary['no_overlap']}")
    print(f"  Training master cave: {dataset_summary['training_master_cave']}")
    print(f"  Evaluation master cave: {dataset_summary['eval_master_cave']}")
    
    # Get the training and evaluation datasets (no overlap)
    training_scene_data = cave_batch_loader.get_training_scene_data()
    eval_scene_data = cave_batch_loader.get_eval_scene_data()
    
    # Create training environment with domain randomization enabled
    training_cave_ids = list(training_scene_data["caves"].keys())
    print(f"\nCreating training environment with domain randomization for {len(training_cave_ids)} caves")
    
    # Create training environment with domain randomization enabled
    env = CaveExplore(
        config=env_cfg, 
        scene_data=training_scene_data, 
        scene_type="training",
        domain_randomization_enabled=True
    )

    # For evaluation, use the same eval cave as the original training if available
    eval_cave_ids = list(eval_scene_data["caves"].keys())
    if original_log_dir and 'selected_eval_cave_id' in loaded_config:
        selected_eval_cave_id = loaded_config['selected_eval_cave_id']
        if selected_eval_cave_id not in eval_cave_ids:
            print(f"Warning: Original eval cave {selected_eval_cave_id} not available, using {eval_cave_ids[0]}")
            selected_eval_cave_id = eval_cave_ids[0]
    else:
        selected_eval_cave_id = eval_cave_ids[0]  # Use first eval cave
        
    eval_env = CaveExplore(
        config=env_cfg, 
        scene_data=eval_scene_data, 
        scene_type="eval",
        domain_randomization_enabled=False
    )
    eval_env.select_cave_environment(selected_eval_cave_id)
    
    print(f"\nEnvironment creation completed:")
    print(f"  Training environment: Domain randomization with {len(training_cave_ids)} caves")
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
    
    # Save configurations including continuation info
    print("Saving configs")
    
    configs = {
        "env_cfg": convert_to_dict(env_cfg),
        "ppo_params": convert_to_dict(ppo_params_input),
        "dataset_summary": dataset_summary,
        "selected_eval_cave_id": selected_eval_cave_id,
        "training_caves_available": training_cave_ids,
        "eval_caves_available": eval_cave_ids,
        "reachbot_model_type": ReachbotModelType.BASIC.name,
        "training_mode": "wrapper_dynamic_caves",
        "domain_randomization": {
            "enabled": 'cave_data_arrays' in locals(),
            "num_caves": cave_data_arrays['all_cave_positions'].shape[0] if 'cave_data_arrays' in locals() else 0,
            "max_boxes_per_cave": cave_data_arrays['all_cave_positions'].shape[1] if 'cave_data_arrays' in locals() else 0,
            "cave_wall_geom_ids_count": len(cave_data_arrays['cave_wall_geom_ids']) if 'cave_data_arrays' in locals() else 0,
        },
        # Add continuation info
        "continuation_info": {
            "is_continuation": original_log_dir is not None,
            "original_log_dir": original_log_dir,
            "initial_step": initial_step,
            "additional_timesteps": ppo_params_input.get("num_timesteps", 0),
            "continuation_timestamp": datetime_str
        } if original_log_dir else {
            "is_continuation": False
        }
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
                    task="Cave Exploration RL Training (Continuation)",
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
        restore_checkpoint_path=checkpoint_path,
        restore_value_fn=True
    )
    
    # Run training (with optional initial parameters for continuation)
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
            task="Cave Exploration RL Training (Continuation)",
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
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    
    print("Setting up rollout for video creation...")
    
    # Rollout parameters
    rng = jax.random.PRNGKey(0)
    n_episodes = 5
    rollout_steps = 5000
    
    # Rollout policy and record simulation
    print(f"Running rollout for {n_episodes} episode(s) with {rollout_steps} steps each...")
    episode_rewards = []
    
    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}")
        episode_rng, rng = jax.random.split(rng)
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
        task="Cave Exploration RL Training (Continuation)",
        duration=run_duration,
        result=f"{training_result}\n{episode_summary}"
    )
    
    print("\n=== TRAINING CONTINUATION AND VIDEO CREATION COMPLETE ===")
    print(f"Log directory: {logdir}")
    print(f"Training duration: {run_duration}")
    print(f"Training result: {training_result}")
    print(f"Episode summary: {episode_summary}")
    
    return episode_rewards

def main():
    """Main function to run the complete training pipeline"""
    print("🚀 Starting Cave Exploration RL Training Continuation Script")
    print(f"📅 Start time: {datetime.now()}")
    print(f"🔥 PID: {os.getpid()} | Thread: {threading.current_thread().ident}")
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration from the specified log directory
        loaded_config = load_config_from_logdir(args.log_dir)
        
        # Find and load the latest checkpoint
        latest_step, latest_checkpoint_path = find_latest_checkpoint(args.log_dir)
        
        # Configure from loaded config
        env_cfg = configure_environment_from_config(loaded_config)
        ppo_params, ppo_training_params = configure_ppo_parameters_from_config(loaded_config, args.additional_timesteps)
        
        print(f"\n=== CONTINUATION SETUP ===")
        print(f"Original log directory: {args.log_dir}")
        print(f"Continuing from step: {latest_step}")
        print(f"Additional timesteps: {args.additional_timesteps}")
        print(f"Latest checkpoint: {latest_checkpoint_path}")
        
        # Call the updated training function with continuation parameters
        env, params, make_inference_fn, logdir, times, total_rewards, total_rewards_std = trainModel(
            ppo_training_params, 
            env_cfg, 
            loaded_config,
            checkpoint_path=latest_checkpoint_path,
            initial_step=latest_step,
            original_log_dir=args.log_dir
        )
        
        # Video creation
        episode_rewards = create_videos(env, params, make_inference_fn, logdir, times, total_rewards, total_rewards_std)
        
        print("✅ All tasks completed successfully!")
        print(f"Original training: {args.log_dir}")
        print(f"Continued training: {logdir}")
        print(f"Training continued from step {latest_step} for {args.additional_timesteps} additional timesteps")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise

# Main function
if __name__ == '__main__':
    main()