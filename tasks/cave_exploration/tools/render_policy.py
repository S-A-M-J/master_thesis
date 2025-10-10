
#!/usr/bin/env python3
"""
Render Policy Script for Cave Exploration

This script renders videos from trained cave exploration policies,
updated to work with the new CaveBatchLoader system.

Usage:
    python render_policy.py --log_dir <path_to_checkpoint> --cave_dir <cave_directory> --n_episodes <num_episodes> --episode_length <steps>

Example:
    python render_policy.py --log_dir /home/user/logs/cave_exploration-2025-09-02_16-31-03 --cave_dir new_caves/005 --n_episodes 5 --episode_length 2000
"""

import argparse
import mujoco
import os
import sys
import imageio
import json
import functools
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# GPU configuration (match run_cave_exploration.py)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.985'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for offscreen rendering


import jax
from jax import numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.io import model
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from ml_collections import config_dict

# Task-specific imports (match run_cave_exploration.py)
from tasks.cave_exploration.training.cave_exploration import CaveExplore, default_config as reachbot_config
from tasks.cave_exploration.environment.env_loader import CaveBatchLoader
from models.model_loader import ReachbotModelType

# JAX configuration
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_traceback_filtering', 'off')
jax.config.update("jax_debug_infs", True)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Render Cave Exploration Policy Videos')
    
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Absolute path to the checkpoint directory')
    parser.add_argument('--cave_dir', type=str, required=True,
                        help='Cave directory name (e.g., new_caves/005)')
    parser.add_argument('--n_episodes', type=int, default=3,
                        help='Number of episodes to render (default: 3)')
    parser.add_argument('--episode_length', type=int, default=3000,
                        help='Maximum number of steps per episode (default: 3000)')
    parser.add_argument('--eval_cave_id', type=int, default=271,
                        help='Evaluation cave ID to use for rendering (default: 271)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.log_dir):
        print(f"Error: Checkpoint directory does not exist: {args.log_dir}")
        exit(1)
    
    config_path = os.path.join(args.log_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        exit(1)
    
    params_path = os.path.join(args.log_dir, 'params')
    if not os.path.exists(params_path):
        print(f"Error: Model parameters not found: {params_path}")
        exit(1)
    
    return args

def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_arguments()
    
    print("🎬 Starting Cave Exploration Policy Rendering")
    print(f"📅 Start time: {datetime.now()}")
    print(f"📁 Checkpoint path: {args.log_dir}")
    print(f"🏔️ Cave directory: {args.cave_dir}")
    print(f"🎬 Number of episodes: {args.n_episodes}")
    print(f"⏱️ Episode length: {args.episode_length}")
    print(f"🔢 Evaluation cave ID: {args.eval_cave_id}")

    print(f"🔢 Evaluation cave ID: {args.eval_cave_id}")

    # Load configuration from the checkpoint
    with open(os.path.join(args.log_dir, 'config.json'), 'r') as f:
        loaded_config = json.load(f)

    print('Rendering cave exploration task result')

    # Get the default environment configuration and update with saved config
    env_cfg = reachbot_config()
    json_env_cfg = config_dict.ConfigDict(loaded_config['env_cfg'])
    env_cfg.update(json_env_cfg)
    env_cfg.randomize_starting_pos = False  # Disable random starting position for rendering

    selected_cave_id = args.eval_cave_id  # Use specified eval cave for rendering  

    # Create CaveBatchLoader to properly load cave environments
    print("Loading cave environments with CaveBatchLoader...")
    cave_batch_loader = CaveBatchLoader(
        env_cfg, 
        ReachbotModelType.BASIC, 
        caves_directory=args.cave_dir,
        eval_cave_index=args.eval_cave_id
    )

    # Print dataset summary
    dataset_summary = cave_batch_loader.get_dataset_summary()
    print(f"\nDataset Summary:")
    print(f"  Total caves: {dataset_summary['total_caves']}")
    print(f"  Training caves: {dataset_summary['training_caves']['count']}")
    print(f"  Evaluation caves: {dataset_summary['eval_caves']['count']}")

    # Get evaluation scene data for rendering (use eval caves for consistent results)
    eval_scene_data = cave_batch_loader.get_eval_scene_data()
    eval_cave_ids = list(eval_scene_data["caves"].keys())

    # Create environment for rendering - use eval environment without domain randomization
    env = CaveExplore(
        config=env_cfg, 
        scene_data=eval_scene_data, 
        scene_type="eval",
        domain_randomization_enabled=False
    )

    # Select a specific cave for rendering
    env.select_cave_environment(selected_cave_id)

    print(f"Environment setup completed:")
    print(f"  Using evaluation cave: {selected_cave_id}")
    print(f"  Available eval caves: {eval_cave_ids}")
    print(f"  Domain randomization: Disabled (for consistent rendering)")

    # Get the PPO configuration (match run_cave_exploration.py)
    ENV_STR = 'Go1JoystickFlatTerrain'
    ppo_params = locomotion_params.brax_ppo_config(ENV_STR)
    ppo_training_params = dict(ppo_params)
    ppo_training_params['num_timesteps'] = 0
    ppo_training_params['num_envs'] = 2 

    print("Network setup:")
    print(f"  Input layer size (observation): {env.observation_size}")
    print(f"  Output layer size (action): {env.action_size}")

    # Network factory setup (match run_cave_exploration.py approach)
    network_factory = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size, 
        action_size=env.action_size
    )

    if "network_factory" in ppo_params:
        if "network_factory" in ppo_training_params:
            del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    # Building the training function for inference setup
    train_fn = functools.partial(
        ppo.train, 
        **dict(ppo_training_params),
        network_factory=network_factory,
    )

    # Building the inference function
    print("Building inference function...")
    make_inference_fn, params, _ = train_fn(
        environment=env,
        num_timesteps=0,
        wrap_env_fn=wrapper.wrap_for_brax_training
    )

    # Load the trained model parameters
    params_path = os.path.join(args.log_dir, 'params')
    print(f"Loading trained parameters from: {params_path}")
    params = model.load_params(params_path)

    # Setup JIT compiled functions for inference
    print("Setting up JIT compiled functions...")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    def render_episodes():
        """Render episodes with the trained policy"""
        print("=== RENDERING EPISODES ===")
        
        # Create a unique timestamp for this rendering session
        render_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        render_session_folder = os.path.join(args.log_dir, f'render_session_{render_timestamp}')
        os.makedirs(render_session_folder, exist_ok=True)
        print(f"Created render session folder: {render_session_folder}")
        
        # Rollout parameters (use command-line arguments)
        rng = jax.random.PRNGKey(0)  # Use seed 0 for reproducible results
        n_episodes = args.n_episodes
        rollout_steps = args.episode_length
        
        # Set this to True to enable detailed logging of state info and rewards
        ENABLE_DETAILED_LOGGING = True
        
        print(f"Running rollout for {n_episodes} episode(s) with {rollout_steps} steps each...")
        print(f"Selected cave for rendering: {selected_cave_id}")
        
        episode_rewards = []
        
        # Initialize logging data if enabled
        if ENABLE_DETAILED_LOGGING:
            detailed_logs = []
            
        for episode in range(n_episodes):
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            
            # Create episode folder
            episode_folder = os.path.join(render_session_folder, f'episode_{episode + 1:02d}')
            os.makedirs(episode_folder, exist_ok=True)
            print(f"Created episode folder: {episode_folder}")
            
            episode_rng, rng = jax.random.split(rng)
            state = jit_reset(episode_rng)
            rollout = [state]  # Reset rollout for each episode
            episode_reward = 0.0
            episode_logs = [] if ENABLE_DETAILED_LOGGING else None
            
            # Log initial state if detailed logging is enabled
            if ENABLE_DETAILED_LOGGING:
                frame_data = {
                    'episode': episode,
                    'step': 0,
                    'reward': float(state.reward),
                    'cumulative_reward': episode_reward,
                    'done': bool(state.done),
                    'cave_id': selected_cave_id,
                    'info': {},
                    'individual_rewards': {}
                }
                # Convert state.info to regular Python types for JSON serialization
                for key, value in state.info.items():
                    # Skip pos_history to reduce log file size
                    if key == 'pos_history':
                        continue
                    if hasattr(value, 'tolist'):  # JAX arrays
                        frame_data['info'][key] = value.tolist()
                    elif hasattr(value, 'item'):  # Scalar arrays
                        frame_data['info'][key] = value.item()
                    else:
                        frame_data['info'][key] = value
                
                # Extract individual reward components from state.metrics
                for key, value in state.metrics.items():
                    if key.startswith('reward/'):
                        reward_name = key[7:]  # Remove 'reward/' prefix
                        if hasattr(value, 'item'):  # Scalar arrays
                            frame_data['individual_rewards'][reward_name] = float(value.item())
                        else:
                            frame_data['individual_rewards'][reward_name] = float(value)
                
                episode_logs.append(frame_data)
            
            for i in range(rollout_steps):
                if i % 500 == 0:
                    print(f"  Step {i}/{rollout_steps}, Current reward: {episode_reward:.3f}")
                    
                act_rng, episode_rng = jax.random.split(episode_rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                
                # Check for numerical issues
                if jp.any(jp.isinf(ctrl)) or jp.any(jp.isnan(ctrl)):
                    print(f"Numerical issue detected in control at step {i}. Stopping rollout.")
                    break
                    
                state = jit_step(state, ctrl)
                
                # Accumulate reward for this episode
                episode_reward += float(state.reward)
                
                # Log detailed state information if enabled
                if ENABLE_DETAILED_LOGGING:
                    frame_data = {
                        'episode': episode,
                        'step': i + 1,
                        'reward': float(state.reward),
                        'cumulative_reward': episode_reward,
                        'done': bool(state.done),
                        'cave_id': selected_cave_id,
                        'info': {},
                        'individual_rewards': {}
                    }
                    # Convert state.info to regular Python types for JSON serialization
                    for key, value in state.info.items():
                        # Skip pos_history to reduce log file size
                        if key == 'pos_history':
                            continue
                        if hasattr(value, 'tolist'):  # JAX arrays
                            frame_data['info'][key] = value.tolist()
                        elif hasattr(value, 'item'):  # Scalar arrays
                            frame_data['info'][key] = value.item()
                        else:
                            frame_data['info'][key] = value
                    
                    # Extract individual reward components from state.metrics
                    for key, value in state.metrics.items():
                        if key.startswith('reward/'):
                            reward_name = key[7:]  # Remove 'reward/' prefix
                            if hasattr(value, 'item'):  # Scalar arrays
                                frame_data['individual_rewards'][reward_name] = float(value.item())
                            else:
                                frame_data['individual_rewards'][reward_name] = float(value)
                    
                    episode_logs.append(frame_data)
                
                if state.done:
                    print(f"Episode {episode + 1} ended at step {i} with reward: {episode_reward:.3f}")
                    break
                    
                rollout.append(state)
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1} completed with {len(rollout)} states and total reward: {episode_reward:.3f}")
            
            if ENABLE_DETAILED_LOGGING:
                detailed_logs.extend(episode_logs)

            # Save detailed logs for this episode if enabled
            if ENABLE_DETAILED_LOGGING:
                log_filename = os.path.join(episode_folder, f'detailed_logs_cave_{selected_cave_id}_reward_{episode_reward:.2f}.json')
                try:
                    with open(log_filename, 'w') as f:
                        json.dump(episode_logs, f, indent=2)
                    print(f"Detailed logs saved to {log_filename}")
                except Exception as e:
                    print(f"Error saving detailed logs: {e}")
                    # Fallback: save as text file
                    txt_filename = os.path.join(episode_folder, f'detailed_logs_cave_{selected_cave_id}_reward_{episode_reward:.2f}.txt')
                    with open(txt_filename, 'w') as f:
                        for frame in episode_logs:
                            f.write(f"Episode: {frame['episode']}, Step: {frame['step']}, "
                                   f"Reward: {frame['reward']:.6f}, Cumulative: {frame['cumulative_reward']:.6f}, "
                                   f"Done: {frame['done']}, Cave: {frame['cave_id']}\n")
                            f.write(f"Info: {frame['info']}\n")
                            if 'individual_rewards' in frame and frame['individual_rewards']:
                                f.write(f"Individual Rewards: {frame['individual_rewards']}\n")
                            f.write("\n")
                    print(f"Detailed logs saved as text to {txt_filename}")

            # Render video (match run_cave_exploration.py approach)
            print("Rendering video...")
            render_every = 1  # Render every frame
            width = 1920      # Full HD width
            height = 1080     # Full HD height

            frames = env.render(rollout[::render_every], camera='track_global', width=width, height=height)
            print(f"Rendered {len(frames)} frames")

            # Save video
            video_path = os.path.join(episode_folder, f'render_cave_{selected_cave_id}_reward_{episode_reward:.1f}.mp4')
            fps = 1.0 / env.dt

            print(f"Saving video to {video_path} at {fps} FPS...")
            imageio.mimsave(video_path, frames, fps=fps)
            print(f"Video saved successfully: Episode {episode + 1}, Reward: {episode_reward:.3f}")
            
            # Create episode summary file
            episode_summary = {
                'episode_number': episode + 1,
                'total_episodes': n_episodes,
                'cave_id': selected_cave_id,
                'episode_reward': episode_reward,
                'rollout_steps': len(rollout),
                'max_steps': rollout_steps,
                'completed_early': len(rollout) < rollout_steps,
                'video_filename': os.path.basename(video_path),
                'log_filename': os.path.basename(log_filename) if ENABLE_DETAILED_LOGGING else None,
                'timestamp': datetime.now().isoformat(),
                'environment_config': {
                    'domain_randomization': False,
                    'randomize_starting_pos': env_cfg.randomize_starting_pos,
                    'scene_type': 'eval'
                }
            }
            
            summary_path = os.path.join(episode_folder, 'episode_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(episode_summary, f, indent=2)
            print(f"Episode summary saved to {summary_path}")

        # Save comprehensive detailed logs for all episodes if enabled
        if ENABLE_DETAILED_LOGGING and 'detailed_logs' in locals():
            comprehensive_log_filename = os.path.join(render_session_folder, f'detailed_logs_all_episodes_cave_{selected_cave_id}.json')
            try:
                with open(comprehensive_log_filename, 'w') as f:
                    json.dump(detailed_logs, f, indent=2)
                print(f"Comprehensive detailed logs for all episodes saved to {comprehensive_log_filename}")
            except Exception as e:
                print(f"Error saving comprehensive detailed logs: {e}")
                # Fallback: save as text file
                txt_filename = os.path.join(render_session_folder, f'detailed_logs_all_episodes_cave_{selected_cave_id}.txt')
                with open(txt_filename, 'w') as f:
                    for frame in detailed_logs:
                        f.write(f"Episode: {frame['episode']}, Step: {frame['step']}, "
                               f"Reward: {frame['reward']:.6f}, Cumulative: {frame['cumulative_reward']:.6f}, "
                               f"Done: {frame['done']}, Cave: {frame['cave_id']}\n")
                        f.write(f"Info: {frame['info']}\n")
                        if 'individual_rewards' in frame and frame['individual_rewards']:
                            f.write(f"Individual Rewards: {frame['individual_rewards']}\n")
                        f.write("\n")
                print(f"Comprehensive detailed logs saved as text to {txt_filename}")

        # Create session summary
        session_summary = {
            'render_timestamp': render_timestamp,
            'cave_id': selected_cave_id,
            'total_episodes': n_episodes,
            'episode_rewards': episode_rewards,
            'average_reward': (sum(episode_rewards)/len(episode_rewards)) if episode_rewards else 0.0,
            'best_episode': {
                'episode_number': (episode_rewards.index(max(episode_rewards)) + 1) if episode_rewards else None,
                'reward': (max(episode_rewards)) if episode_rewards else None
            },
            'worst_episode': {
                'episode_number': (episode_rewards.index(min(episode_rewards)) + 1) if episode_rewards else None,
                'reward': (min(episode_rewards)) if episode_rewards else None
            },
            'environment_config': {
                'domain_randomization': False,
                'randomize_starting_pos': env_cfg.randomize_starting_pos,
                'scene_type': 'eval'
            },
            'render_config': {
                'rollout_steps': rollout_steps,
                'render_every': 1,
                'width': 1920,
                'height': 1080,
                'camera': 'track_global'
            }
        }

        session_summary_path = os.path.join(render_session_folder, 'render_session_summary.json')
        with open(session_summary_path, 'w') as f:
            json.dump(session_summary, f, indent=2)
        print(f"Render session summary saved to {session_summary_path}")

        # Print summary of all episodes (match run_cave_exploration.py)
        print("\n=== EPISODE REWARD SUMMARY ===")
        for i, reward in enumerate(episode_rewards):
            print(f"Episode {i + 1}: {reward:.3f}")
        if episode_rewards:
            print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.3f}")
            print(f"Best episode: {episode_rewards.index(max(episode_rewards)) + 1} with reward {max(episode_rewards):.3f}")
            print(f"Worst episode: {episode_rewards.index(min(episode_rewards)) + 1} with reward {min(episode_rewards):.3f}")
        print(f"Cave used for rendering: {selected_cave_id}")

        return episode_rewards

    # Run the rendering
    if __name__ == '__main__':
        print("🎬 Starting Cave Exploration Policy Rendering")
        print(f"📅 Start time: {datetime.now()}")
        
        try:
            episode_rewards = render_episodes()
            print("✅ Rendering completed successfully!")
            
        except Exception as e:
            print(f"❌ Error occurred during rendering: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    main()