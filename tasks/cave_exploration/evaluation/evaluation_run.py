#!/usr/bin/env python3
"""
Efficient evaluation runner for Cave Exploration policies.

Features:
- Loads a trained policy from a run folder (or a specific checkpoint) once.
- Builds env and JITs reset/step/inference a single time.
- Iterates all eval caves and runs N episodes per cave without rebuilding the env.
- Saves detailed per-episode metrics per cave into the same run folder.

Usage example:
  python tasks/cave_exploration/evaluation_run.py \
	--log_dir logs/cave_exploration-2025-08-19_21-01-27 \
	--caves_directory new_caves/001 \
	--episodes_per_cave 3 --max_steps 3000

Optionally use a specific checkpoint (orbax saved directory) instead of final params:
  --checkpoint logs/.../checkpoints/20000000
"""

import os
import sys
import json
import argparse
import functools
from datetime import datetime
from tqdm import tqdm

# Always flush prints
print = functools.partial(print, flush=True)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# GPU/JAX runtime config: keep memory growth friendly and avoid re-preallocation
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
from jax import numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.io import model
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from ml_collections import config_dict

from flax.training import orbax_utils
from orbax import checkpoint as ocp

# Task-specific
from tasks.cave_exploration.training.cave_exploration import CaveExplore, default_config as reachbot_config
from tasks.cave_exploration.environment.env_loader import CaveBatchLoader
from models.model_loader import ReachbotModelType
from tasks.cave_exploration.environment.domain_randomize import prepare_cave_data_arrays


def parse_args():
	p = argparse.ArgumentParser(description="Evaluate a trained cave exploration policy across all eval caves.")
	p.add_argument('--log_dir', required=True, help='Path to the training run folder (contains config.json, params/, checkpoints/)')
	p.add_argument('--checkpoint', help='Optional path to a specific checkpoint directory (orbax) to load params from')
	p.add_argument('--caves_directory', help='Override caves directory (if not provided, tries config.json field)')
	p.add_argument('--episodes_per_cave', type=int, default=3, help='Number of episodes per cave')
	p.add_argument('--max_steps', type=int, default=4000, help='Max steps per episode')
	p.add_argument('--seed', type=int, default=0, help='Base RNG seed')
	return p.parse_args()


def load_run_config(log_dir: str) -> dict:
	cfg_path = os.path.join(log_dir, 'config.json')
	if not os.path.exists(cfg_path):
		raise FileNotFoundError(f'config.json not found in {log_dir}')
	with open(cfg_path, 'r') as f:
		return json.load(f)


def load_policy_params(log_dir: str, checkpoint: str | None, make_inference_fn):
	"""Load params either from a specific orbax checkpoint dir or from params/ folder."""
	if checkpoint:
		# Orbax checkpoint restore
		print(f"Loading params from checkpoint: {checkpoint}")
		checkpointer = ocp.PyTreeCheckpointer()
		params = checkpointer.restore(os.path.abspath(checkpoint))
		return params
	# Fallback: final params saved via brax.io.model
	params_path = os.path.join(log_dir, 'params')
	if not os.path.exists(params_path):
		raise FileNotFoundError(f'No params found at {params_path}. Provide --checkpoint to restore.')
	print(f"Loading params from: {params_path}")
	return model.load_params(params_path)


def to_py(obj):
	"""Recursively convert JAX arrays and numpy scalars to Python types for JSON."""
	if hasattr(obj, 'tolist'):
		return obj.tolist()
	if hasattr(obj, 'item'):
		try:
			return obj.item()
		except Exception:
			pass
	if isinstance(obj, dict):
		return {k: to_py(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [to_py(v) for v in obj]
	return obj


def main():
	args = parse_args()

	print(f"Starting evaluation run with PID {os.getpid()}")
	print(f"Run folder: {args.log_dir}")
	if args.checkpoint:
		print(f"Using checkpoint: {args.checkpoint}")

	# Record start time
	eval_start = datetime.now()
	print(f"Start time: {eval_start.isoformat()}")

	# Load training config to reproduce env + network settings
	loaded_config = load_run_config(args.log_dir)

	# Build env config
	env_cfg = reachbot_config()
	json_env_cfg = config_dict.ConfigDict(loaded_config.get('env_cfg', {}))
	env_cfg.update(json_env_cfg)
	env_cfg.stickiness_config.stickiness_force = 0.0  # Ensure stickiness is on for eval
	#env_cfg.noise_config.level = 1.0
	#env_cfg.noise_config.scales.update(
	#	joint_pos=0.0,
	#	joint_vel=0.0,
	#	gyro=0.0,
	#	gravity=0.0,
	#	linvel=0.0,
	#	#lidar=0.0  # disable lidar noise for eval
	#)
	# Caves directory resolution
	caves_dir = args.caves_directory or loaded_config.get('caves_directory')
	if not caves_dir:
		raise ValueError('caves_directory not provided and not found in config.json')
	print(f"Caves directory: {caves_dir}")

	# Prepare loader and eval scene data (no overlap with training)
	print("Loading CaveBatchLoader and eval scene data...")
	cave_batch_loader = CaveBatchLoader(env_cfg, ReachbotModelType.BASIC, caves_directory=caves_dir)
	eval_scene_data = cave_batch_loader.get_eval_scene_data()
	eval_cave_ids = list(eval_scene_data["caves"].keys())
	print(f"Eval caves: {len(eval_cave_ids)}")

	# Prepare eval cave arrays used to move geometries between cave switches
	prepared = prepare_cave_data_arrays(cave_batch_loader)
	eval_arrays = prepared['eval']
	arr_positions = eval_arrays['all_cave_positions']  # [num_eval_caves, max_boxes, 3]
	arr_counts = eval_arrays['cave_box_counts']        # [num_eval_caves]
	geom_ids = eval_arrays['cave_wall_geom_ids']       # [<= max_boxes]
	arr_cave_ids = eval_arrays['cave_ids']             # python list of cave ids
	# Map cave_id -> index into arrays
	cave_id_to_idx = {cid: i for i, cid in enumerate(arr_cave_ids)}

	# Create a single eval env, no domain randomization for consistency
	env = CaveExplore(
		config=env_cfg,
		scene_data=eval_scene_data,
		scene_type='eval',
		domain_randomization_enabled=False,
	)
	env.select_cave_environment(cave_batch_loader.get_master_cave_id("eval"))

	# Build networks once based on env sizes
	ENV_STR = 'Go1JoystickFlatTerrain'
	ppo_params = locomotion_params.brax_ppo_config(ENV_STR)
	print(f"Obs size: {env.observation_size}, Act size: {env.action_size}")
	network_factory = ppo_networks.make_ppo_networks(
		observation_size=env.observation_size,
		action_size=env.action_size,
	)
	if "network_factory" in ppo_params:
		network_factory = functools.partial(
			ppo_networks.make_ppo_networks,
			**ppo_params.network_factory
		)

	# Create train fn only to get make_inference_fn signature
	ppo_training_params = dict(ppo_params)
	ppo_training_params['num_timesteps'] = 0
	ppo_training_params['num_envs'] = 2

	# Avoid passing network_factory twice if present in config
	if 'network_factory' in ppo_training_params:
		del ppo_training_params['network_factory']

	train_fn = functools.partial(
		ppo.train,
		**ppo_training_params,
		network_factory=network_factory,
	)

	# Build inference function graph (num_timesteps=0 => no training)
	print("Building inference function...")
	make_inference_fn, _, _ = train_fn(
		environment=env,
		num_timesteps=0,
		wrap_env_fn=wrapper.wrap_for_brax_training,
	)

	# Load params from checkpoint or final params
	params = load_policy_params(args.log_dir, args.checkpoint, make_inference_fn)

	# Build deterministic inference once; reset/step will be re-jitted per cave to pick up geom changes
	inference_fn = make_inference_fn(params, deterministic=True)
	jit_inference_fn = jax.jit(inference_fn)

	# Prepare evaluation output dir inside the run folder
	eval_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	eval_root = os.path.join(args.log_dir, f'eval_session_{eval_ts}')
	os.makedirs(eval_root, exist_ok=True)
	print(f"Results will be saved to: {eval_root}")

	rng = jax.random.PRNGKey(args.seed)

	# No global warmup for reset/step; we re-jit per cave after switching geometry

	# Summary tracking
	session_summary = {
		'run_dir': args.log_dir,
		'checkpoint': args.checkpoint,
		'caves_directory': caves_dir,
		'episodes_per_cave': args.episodes_per_cave,
		'max_steps': args.max_steps,
		'eval_timestamp': eval_ts,
		'start_time': eval_start.isoformat(),
		'caves': [],
	}

	for idx, cave_id in enumerate(tqdm(eval_cave_ids, desc="Caves")):
		print(f"[{idx+1}/{len(eval_cave_ids)}] Evaluating cave {cave_id}...")
		env.select_cave_environment(cave_id)

		# Move geometries for this cave using eval arrays (no changes to CaveExplore itself)
		if cave_id not in cave_id_to_idx:
			print(f"  Warning: cave_id {cave_id} not found in prepared arrays; skipping geometry move.")
		else:
			cidx = cave_id_to_idx[cave_id]
			model = env.mjx_model
			geom_pos = model.geom_pos
			num_geoms = int(geom_ids.shape[0])
			num_boxes = int(arr_counts[cidx]) if hasattr(arr_counts[cidx], 'item') else int(arr_counts[cidx])
			# Set positions for present boxes
			if num_boxes > 0:
				positions = arr_positions[cidx, :num_boxes]
				geom_pos = geom_pos.at[geom_ids[:num_boxes]].set(positions)
			# Hide remaining boxes underground
			if num_boxes < num_geoms:
				hide = jp.array([0.0, 0.0, -1000.0])
				geom_pos = geom_pos.at[geom_ids[num_boxes:]].set(hide)
			# Replace model with updated geom positions
			env._scene_data["mjx_model"] = model.tree_replace({"geom_pos": geom_pos})

		# Re-JIT reset/step after cave switch to capture geometry updates
		jit_reset = jax.jit(env.reset)
		jit_step = jax.jit(env.step)

		cave_dir = os.path.join(eval_root, f'cave_{cave_id}')
		os.makedirs(cave_dir, exist_ok=True)

		cave_result = {
			'cave_id': cave_id,
			'episodes': [],
		}

		for ep in tqdm(range(args.episodes_per_cave), desc=f"Cave {cave_id} episodes", leave=False):
			ep_rng, rng = jax.random.split(rng)
			state = jit_reset(ep_rng)
			cumulative_reward = 0.0

			episode_log = {
				'episode_index': ep + 1,
				'steps': [],
				'terminated_early': False,
			}

			for step in range(args.max_steps):
				if step % 500 == 0 and step > 0:
					print(f"  Cave {cave_id}: step {step}/{args.max_steps}, cum_rew={cumulative_reward:.2f}")

				act_rng, ep_rng = jax.random.split(ep_rng)
				ctrl, _ = jit_inference_fn(state.obs, act_rng)

				# Basic numerical safety
				if jp.any(jp.isinf(ctrl)) or jp.any(jp.isnan(ctrl)):
					print(f"  Numerical issue in control at step {step}; aborting episode.")
					episode_log['terminated_early'] = True
					break

				state = jit_step(state, ctrl)
				cumulative_reward += float(state.reward)

				# Capture per-step metrics
				step_record = {
					'step': step,
					'reward': float(state.reward),
					'cumulative_reward': cumulative_reward,
					'done': bool(state.done),
					'metrics': to_py(state.metrics),
					'info': to_py({k: v for k, v in state.info.items() if k != 'pos_history'}),
				}
				episode_log['steps'].append(step_record)

				if state.done:
					break

			episode_log['final_cumulative_reward'] = cumulative_reward

			# Persist per-episode log (one file per episode per cave)
			ep_path = os.path.join(cave_dir, f'episode_{ep+1:02d}.json')
			with open(ep_path, 'w') as f:
				json.dump(episode_log, f, indent=2)
			print(f"  Saved episode {ep+1} log: {ep_path}")

			cave_result['episodes'].append({
				'episode_index': ep + 1,
				'final_cumulative_reward': cumulative_reward,
				'steps': len(episode_log['steps']),
				'terminated_early': episode_log['terminated_early'],
				'log_file': os.path.basename(ep_path),
			})

		# Cave-level summary file
		cave_summary_path = os.path.join(cave_dir, 'cave_summary.json')
		with open(cave_summary_path, 'w') as f:
			json.dump(cave_result, f, indent=2)
		print(f"Saved cave summary: {cave_summary_path}")

		session_summary['caves'].append({
			'cave_id': cave_id,
			'episodes': cave_result['episodes'],
			'dir': os.path.basename(cave_dir),
		})


	# Finish timestamps and duration
	eval_end = datetime.now()
	duration = eval_end - eval_start
	session_summary['end_time'] = eval_end.isoformat()
	session_summary['duration_seconds'] = duration.total_seconds()
	session_summary['duration'] = str(duration)
	session_summary['env_config'] = to_py(env_cfg.to_dict())

	# Write session summary
	session_summary_path = os.path.join(eval_root, 'session_summary.json')
	with open(session_summary_path, 'w') as f:
		json.dump(session_summary, f, indent=2)
	print(f"Finished at: {eval_end.isoformat()} (duration: {duration})")
	print(f"Evaluation complete. Session summary: {session_summary_path}")


if __name__ == '__main__':
	main()

