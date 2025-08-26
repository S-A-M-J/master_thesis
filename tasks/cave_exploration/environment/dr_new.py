

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

# Import the torso body ID constant
from ..common import reachbot_constants as consts


def create_cave_domain_randomizer(cave_data_arrays: dict, max_boxes: int = 7500):
    """Create a JAX-compilable domain randomization function for cave environments.
    
    Args:
        cave_data_arrays: Dictionary containing:
            - 'all_cave_positions': Array of shape [num_caves, max_boxes, 3] with cave box positions
            - 'cave_box_counts': Array of shape [num_caves] with actual number of boxes per cave
            - 'cave_wall_geom_ids': Array of shape [max_boxes] with geom IDs for cave walls
        max_boxes: Maximum number of boxes to support
    
    Returns:
        A JAX-compilable domain randomization function
    """
    
    all_cave_positions = cave_data_arrays['all_cave_positions']  # [num_caves, max_boxes, 3]
    cave_box_counts = cave_data_arrays['cave_box_counts']        # [num_caves]
    cave_wall_geom_ids = cave_data_arrays['cave_wall_geom_ids']  # [max_boxes]
    
    def domain_randomize(model: mjx.Model, rng: jax.Array):
        """JAX-compilable domain randomization with standard Brax signature that randomly selects caves."""
        
        @jax.vmap
        def rand_dynamics(rng):
            # Split RNG for cave selection and domain randomization
            rng, cave_rng = jax.random.split(rng)
            
            # Select random cave for this environment reset
            num_caves = all_cave_positions.shape[0]
            cave_idx = jax.random.randint(cave_rng, (), 0, num_caves)
            
            # Get cave-specific data using cave_idx
            box_positions = all_cave_positions[cave_idx]  # [max_boxes, 3]
            num_wanted_boxes = cave_box_counts[cave_idx]  # scalar
            
            # Update box positions
            geom_pos = model.geom_pos
            num_cave_geoms = cave_wall_geom_ids.shape[0]
            
            # Create mask for boxes to place (avoid dynamic indexing with traced values)
            box_indices = jp.arange(num_cave_geoms)
            boxes_to_place_mask = box_indices < num_wanted_boxes
            
            # Create new positions for all cave boxes   
            new_positions = jp.where(
                boxes_to_place_mask[:, None],
                box_positions[:num_cave_geoms],  # Use cave positions for wanted boxes
                jp.array([0.0, 0.0, -1000.0])   # Hide unwanted boxes underground
            )
            
            # Update all cave wall geometry positions at once
            geom_pos = geom_pos.at[cave_wall_geom_ids].set(new_positions)
            
            # Store cave_idx in the last box geometry position (x, y, z = cave_idx, cave_idx, cave_idx)
            # This allows the environment to retrieve which cave was selected
            last_box_geom_id = cave_wall_geom_ids[-1]  # Use the last box geometry
            cave_idx_float = cave_idx.astype(jp.float32)
            cave_info_position = jp.array([cave_idx_float, cave_idx_float, cave_idx_float])
            geom_pos = geom_pos.at[last_box_geom_id].set(cave_info_position)
            
            #jax.debug.print("Domain randomization applied with {num_boxes} boxes", num_boxes=num_wanted_boxes)
            
            return geom_pos
        
        # Apply vectorized randomization function
        geom_pos = rand_dynamics(rng)
        
        # Set up in_axes for proper batching - geom_pos is batched along first dimension
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({
            "geom_pos": 0,  # Batch over first dimension for geom_pos
        })
        
        model = model.tree_replace({
            "geom_pos": geom_pos,
        })
        
        return model, in_axes
    
    return domain_randomize


def prepare_cave_data_arrays(mj_model, cave_batch_loader, max_boxes: int = 7500):
    """Prepare cave data in JAX-compatible arrays for domain randomization.
    
    Args:
        mj_model: The MuJoCo model
        cave_batch_loader: The CaveBatchLoader instance
        max_boxes: Maximum number of boxes to support across all caves
        
    Returns:
        Dictionary with JAX arrays ready for use in domain randomization
    """
    # Get training scene data from cave_batch_loader
    trainings_scene_data = cave_batch_loader.training_scene
    caves = trainings_scene_data["caves"]
    
    num_caves = len(caves)
    cave_ids = list(caves.keys())
    
    # Initialize arrays
    all_cave_positions = jp.zeros((num_caves, max_boxes, 3))
    cave_box_counts = jp.zeros(num_caves, dtype=jp.int32)
    
    # Get master cave geom IDs (assuming they're consistent)
    master_cave_geom_ids = []
    master_cave_id = trainings_scene_data["master_cave_id"]

    for i in range(mj_model.ngeom):
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and geom_name.startswith(f"cave_wall_box_{master_cave_id}_"):
            master_cave_geom_ids.append(i)
    
    cave_wall_geom_ids = jp.array(master_cave_geom_ids[:max_boxes])
    
    # Fill in cave data
    for cave_idx, cave_id in enumerate(cave_ids):
        cave_data = caves[cave_id]
        boxes = cave_data["boxes"]  # Use "boxes" key instead of "voxel_positions"
        
        num_boxes = min(len(boxes), max_boxes)
        cave_box_counts = cave_box_counts.at[cave_idx].set(num_boxes)
        
        if num_boxes > 0:
            # Convert position dictionaries to [x, y, z] arrays
            position_list = []
            for box in boxes[:num_boxes]:
                if isinstance(box, dict) and "position" in box:
                    # Extract position from box dict structure: box["position"]["x/y/z"]
                    pos = box["position"]
                    position_list.append([pos.get('x'), pos.get('y'), pos.get('z')])
                elif isinstance(box, dict):
                    # Direct position dict with 'x', 'y', 'z' keys
                    position_list.append([box.get('x'), box.get('y'), box.get('z')])
                else:
                    # Assume it's already in list/array format
                    position_list.append(box)
            
            positions_array = jp.array(position_list)
            all_cave_positions = all_cave_positions.at[cave_idx, :num_boxes].set(positions_array)
    
    return {
        'all_cave_positions': all_cave_positions,
        'cave_box_counts': cave_box_counts,
        'cave_wall_geom_ids': cave_wall_geom_ids,
        'cave_ids': cave_ids,
    }

