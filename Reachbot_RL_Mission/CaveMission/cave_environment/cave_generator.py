import numpy as np
import trimesh
import os
import json
import argparse
import time
from trimesh.voxel import creation

def create_dent(theta, radius, dent_center=np.pi, dent_width=np.pi/2, dent_depth=0.8):
    """Create a dent in the circular cross-section"""
    # Normal circular radius
    r = np.ones_like(theta) * radius
    
    # Add gaussian-shaped dent
    angle_diff = np.abs(theta - dent_center)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # Consider circular nature
    dent = dent_depth * np.exp(-(angle_diff**2) / (2 * (dent_width/2)**2))
    r -= dent
    
    return r

def generate_cave_tunnel(curve_intensity=0.5):
    # Parameters
    length = 20.0  # meters
    radius = 2.0  # meters
    num_segments = 10
    points_per_circle = 12
    offset_magnitude = 2
    num_dents = 3  # Number of dents to add
    
    # Randomly choose segments for dents (excluding first and last)
    available_segments = list(range(1, num_segments - 1))  # All segments except first and last
    dented_segments = np.random.choice(
        available_segments, 
        size=min(num_dents, len(available_segments)), 
        replace=False
    )
    
    print(f"Adding dents to segments: {dented_segments}")
    
    # Generate centerline with slight random offsets
    t = np.linspace(0, length, num_segments)
    centerline = np.zeros((num_segments, 3))
    centerline[:, 0] = t  # Base X coordinates
    
    # Add a smooth curve in the Y direction (parabolic shape)
    curve_direction = np.random.choice([-1, 1])  # Random direction
    centerline[:, 1] = curve_direction * curve_intensity * (t - t[0]) * (t - t[-1]) / 25
    
    # Add random offsets to Y and Z coordinates
    # Using smooth transitions with cumsum to avoid sharp changes
    random_offsets_y = np.random.uniform(-offset_magnitude, offset_magnitude, num_segments)
    random_offsets_z = np.random.uniform(-offset_magnitude, offset_magnitude, num_segments)
    
    # Smooth out the offsets and add them to the curve
    centerline[:, 1] += np.cumsum(random_offsets_y) * 0.1  # Y offset (reduced factor to not overwhelm the curve)
    centerline[:, 2] = np.cumsum(random_offsets_z) * 0.3  # Z offset
    
    # Reset first position to avoid drift at the start
    centerline[0] = [0, 0, 0]
    
    # Generate vertices around centerline
    vertices = []
    for i in range(num_segments):
        theta = np.linspace(0, 2*np.pi, points_per_circle, endpoint=False)
        
        # Create cross-section, with potential dent
        if i in dented_segments:
            # Random dent parameters with more variation
            dent_center = np.random.uniform(0, 2*np.pi)  # Random angle
            dent_width = np.random.uniform(np.pi/6, np.pi/2)  # Random width
            dent_depth = np.random.uniform(0.4, 1.0)  # Random depth
            
            # Get radii with dent
            r = create_dent(theta, radius, dent_center, dent_width, dent_depth)
            
            circle = np.column_stack([
                np.zeros_like(theta),
                r * np.cos(theta),
                r * np.sin(theta)
            ])
        else:
            # Regular circular cross-section
            circle = np.column_stack([
                np.zeros_like(theta),
                radius * np.cos(theta),
                radius * np.sin(theta)
            ])
        
        # Move circle to position along centerline
        circle += centerline[i]
        vertices.append(circle)
    
    vertices = np.vstack(vertices)
    
    # Create faces by connecting adjacent circles with proper winding order
    faces = []
    for i in range(num_segments - 1):
        for j in range(points_per_circle):
            # Get indices for current quad
            v0 = i * points_per_circle + j
            v1 = i * points_per_circle + (j + 1) % points_per_circle
            v2 = (i + 1) * points_per_circle + (j + 1) % points_per_circle
            v3 = (i + 1) * points_per_circle + j
            
            # Create triangles (only one side needed)
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    
    faces = np.array(faces)
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return mesh

def voxelize_mesh(mesh, voxel_size=0.1):
    """Convert mesh to voxels with specified resolution"""
    # Skip mesh repair to avoid networkx dependency issues
    # try:
    #     # Ensure mesh is watertight for proper voxelization
    #     mesh.fill_holes()
    #     mesh.fix_normals()
    # except Exception as e:
    #     print(f"Warning: Could not repair mesh: {e}")
    #     print("Continuing with voxelization of unrepaired mesh.")
    
    try:
        # Voxelize the mesh
        voxel_grid = mesh.voxelized(pitch=voxel_size)
        
        # Get filled voxels
        voxels = voxel_grid.sparse_indices
        
        # Return both the voxel grid and points
        return voxel_grid, voxels
    except Exception as e:
        print(f"Error during voxelization: {e}")
        print("Returning empty voxel grid")
        # Return empty data if voxelization fails
        return None, np.array([])

def merge_adjacent_voxels(voxels, voxel_size):
    """
    Merge adjacent voxels into larger boxes when they form complete surfaces
    Returns a list of boxes as (position, size) tuples
    """
    if len(voxels) == 0:
        return []
    
    # Convert voxels to a set for quick lookup
    voxel_set = set(tuple(v) for v in voxels)
    processed = set()
    merged_boxes = []
    
    # Process all voxels
    for voxel in voxels:
        voxel_tuple = tuple(voxel)
        if voxel_tuple in processed:
            continue
            
        # Try to merge in all three axis directions
        for axis in range(3):  # 0=x, 1=y, 2=z
            # Find the two other axes
            other_axes = [i for i in range(3) if i != axis]
            
            # Start with current voxel
            min_pos = list(voxel)
            max_pos = list(voxel)
            
            # Expand in the positive direction of the axis
            keep_expanding = True
            while keep_expanding:
                # Move one step in the axis direction
                test_pos = list(max_pos)
                test_pos[axis] += 1
                
                # Check if there's a complete plane of voxels in the other two dimensions
                # For the current slice position
                plane_complete = True
                
                # Define the base position for this slice
                base_pos = list(min_pos)
                base_pos[axis] = test_pos[axis]
                
                # Check if all voxels in the slice exist
                for i in range(max_pos[other_axes[0]] - min_pos[other_axes[0]] + 1):
                    for j in range(max_pos[other_axes[1]] - min_pos[other_axes[1]] + 1):
                        check_pos = list(base_pos)
                        check_pos[other_axes[0]] = min_pos[other_axes[0]] + i
                        check_pos[other_axes[1]] = min_pos[other_axes[1]] + j
                        
                        if tuple(check_pos) not in voxel_set:
                            plane_complete = False
                            break
                    
                    if not plane_complete:
                        break
                
                if plane_complete:
                    max_pos[axis] += 1
                else:
                    keep_expanding = False
            
            # Now try to expand in other dimensions
            for expand_axis in other_axes:
                # Try expanding in positive direction
                keep_expanding = True
                while keep_expanding:
                    test_pos = list(max_pos)
                    test_pos[expand_axis] += 1
                    
                    # Check if the entire slice exists
                    slice_complete = True
                    for i in range(min_pos[axis], max_pos[axis] + 1):
                        for j in range(min_pos[other_axes[1-other_axes.index(expand_axis)]], 
                                      max_pos[other_axes[1-other_axes.index(expand_axis)]] + 1):
                            check_pos = [0, 0, 0]
                            check_pos[axis] = i
                            check_pos[expand_axis] = test_pos[expand_axis]
                            check_pos[other_axes[1-other_axes.index(expand_axis)]] = j
                            
                            if tuple(check_pos) not in voxel_set:
                                slice_complete = False
                                break
                    
                    if slice_complete:
                        max_pos[expand_axis] += 1
                    else:
                        keep_expanding = False
            
            # Mark all voxels in the box as processed
            for x in range(min_pos[0], max_pos[0] + 1):
                for y in range(min_pos[1], max_pos[1] + 1):
                    for z in range(min_pos[2], max_pos[2] + 1):
                        processed.add((x, y, z))
            
            # Calculate box position and size
            # Position is the center of the box
            position = [(min_pos[i] + max_pos[i]) / 2 * voxel_size + voxel_size/2 for i in range(3)]
            # Size is half the dimensions (MuJoCo convention)
            size = [(max_pos[i] - min_pos[i] + 1) / 2 * voxel_size for i in range(3)]
            
            merged_boxes.append((position, size))
            
            # We've created a box, no need to try other axes for this starter voxel
            break
    
    return merged_boxes

def save_mujoco_boxes(voxels, voxel_size, filename):
    """Save voxels as MuJoCo boxes in XML format, merging adjacent voxels into larger boxes"""
    # Merge adjacent voxels
    merged_boxes = merge_adjacent_voxels(voxels, voxel_size)
    print(f"Reduced from {len(voxels)} individual voxels to {len(merged_boxes)} merged boxes")
    
    with open(filename, 'w') as f:
        f.write('<body name="cave_voxelized" pos="0 0 0">\n')
        
        # Add each merged box as a geom
        for i, (position, size) in enumerate(merged_boxes):
            x, y, z = position
            sx, sy, sz = size
             # Contype and conaffinity are set to 2 and 1 respectively to avoid collisions with other cave wall primitives
            f.write(f'    <geom name="box_{i}" type="box" pos="{x} {y} {z}" size="{sx} {sy} {sz}" material="cave_wall" contype="2" conaffinity="1"/>\n')
        
        f.write('</body>')


def save_mujoco_spheres(voxels, voxel_size, filename):
    """Save voxels as MuJoCo spheres in XML format"""
    with open(filename, 'w') as f:
        f.write('<body name="cave_voxelized" pos="0 0 0">\n')
        
        # Add each voxel as a sphere
        for i, voxel in enumerate(voxels):
            # Convert voxel coordinates to position
            # Voxel coordinates are in grid space, convert to world space
            # by multiplying with voxel size
            x, y, z = voxel
            # Contype and conaffinity are set to 2 and 1 respectively to avoid collisions with other cave wall primitives
            f.write(f'    <geom name="sphere_{i}" type="sphere" pos="{x * voxel_size} {y * voxel_size} {z * voxel_size}" size="{voxel_size / 2}" material="cave_wall" contype="2" conaffinity="1"/>\n')
        
        f.write('</body>')

def create_cave(cave_id, output_dir, curve_intensity=0.5):
    """Create a single cave and save to the specified directory"""
    # Parameters to save
    cave_params = {}
    
    # Parameters for this cave
    length = 20.0  # meters
    radius = 2.0  # meters
    num_segments = 10
    points_per_circle = 12
    offset_magnitude = 2
    num_dents = 3  # Number of dents to add
    
    # Save generation parameters
    cave_params["generation"] = {
        "cave_id": cave_id,
        "length": length,
        "radius": radius,
        "num_segments": num_segments,
        "points_per_circle": points_per_circle,
        "offset_magnitude": offset_magnitude,
        "num_dents": num_dents,
        "curve_intensity": curve_intensity
    }
    
    # Generate the original mesh with the parameters
    cave_mesh = generate_cave_tunnel(curve_intensity)
    print(f"Cave {cave_id} - Original mesh - Vertex count: {len(cave_mesh.vertices)}")
    print(f"Cave {cave_id} - Original mesh - Face count: {len(cave_mesh.faces)}")
    
    # Save mesh stats
    cave_params["original_mesh"] = {
        "vertex_count": len(cave_mesh.vertices),
        "face_count": len(cave_mesh.faces),
        "is_watertight": cave_mesh.is_watertight
    }
    
    try:
        # Save original mesh
        mesh_path = os.path.join(output_dir, "cave_tunnel_original.stl")
        print(f"Saving original mesh to {mesh_path}")
        cave_mesh.export(mesh_path)
    except Exception as e:
        print(f"Error saving original mesh: {e}")
    
    # Voxelize the mesh with 0.1m resolution
    voxel_size = 0.1  # meters
    voxel_grid, voxels = voxelize_mesh(cave_mesh, voxel_size)
    print(f"Cave {cave_id} - Voxelized mesh - Number of voxels: {len(voxels)}")
    
    # Save voxelization stats
    cave_params["voxelization"] = {
        "voxel_size": voxel_size,
        "voxel_count": len(voxels)
    }
    
    # Calculate target position based on voxel data
    if len(voxels) > 0:
        # Convert voxel indices to world coordinates
        world_voxels = voxels * voxel_size
        
        # Find the voxel with maximum x value
        max_x = np.max(world_voxels[:, 0])
        max_x_voxels = world_voxels[world_voxels[:, 0] == max_x]
        
        # Find y range of voxels at max x position
        max_y = np.max(max_x_voxels[:, 1])
        min_y = np.min(max_x_voxels[:, 1])
        mid_y = (max_y + min_y) / 2
        
        # Find lowest z value among voxels at max x
        min_z = np.min(max_x_voxels[:, 2])
        target_z = min_z + 1.0  # 1m above the lowest z at max x
        
        # Save target position
        cave_params["target_pos"] = {
            "x": float(max_x),
            "y": float(mid_y),
            "z": float(target_z)
        }
        
        print(f"Cave {cave_id} - Target position: x={max_x:.2f}, y={mid_y:.2f}, z={target_z:.2f}")
    else:
        print(f"Cave {cave_id} - Could not calculate target position (no voxels)")
        cave_params["target_pos"] = {
            "x": length,  # Default to end of tunnel
            "y": 0.0,
            "z": 1.0
        }
    
    # Convert voxels to a new mesh if voxel_grid is not None
    if voxel_grid is not None:
        voxel_mesh = voxel_grid.as_boxes()
        print(f"Cave {cave_id} - Voxel mesh - Vertex count: {len(voxel_mesh.vertices)}")
        print(f"Cave {cave_id} - Voxel mesh - Face count: {len(voxel_mesh.faces)}")
        
        # Save voxel mesh stats
        cave_params["voxel_mesh"] = {
            "vertex_count": len(voxel_mesh.vertices),
            "face_count": len(voxel_mesh.faces)
        }
        
        # Save voxelized mesh as STL
        voxel_mesh.export(os.path.join(output_dir, "cave_tunnel_voxels.stl"))
    else:
        print(f"Cave {cave_id} - Skipping voxel mesh creation due to voxelization failure")
        cave_params["voxel_mesh"] = {
            "vertex_count": 0,
            "face_count": 0
        }
    
    # Create MuJoCo XML with boxes
    merged_boxes = merge_adjacent_voxels(voxels, voxel_size)
    print(f"Cave {cave_id} - Reduced from {len(voxels)} individual voxels to {len(merged_boxes)} merged boxes")
    
    # Save MuJoCo box stats
    cave_params["mujoco_boxes"] = {
        "original_box_count": len(voxels),
        "merged_box_count": len(merged_boxes)
    }
    
    save_mujoco_boxes(voxels, voxel_size, os.path.join(output_dir, "cave_tunnel_boxes.xml"))

    # Create MuJoCo XML with spheres instead of boxes
    save_mujoco_spheres(voxels, voxel_size, os.path.join(output_dir, "cave_tunnel_spheres.xml"))
    
    # Save all parameters and stats to JSON file
    json_path = os.path.join(output_dir, "cave_tunnel_info.json")
    with open(json_path, 'w') as f:
        json.dump(cave_params, f, indent=4)
    
    print(f"Cave {cave_id} information saved to {json_path}")
    
    return cave_params

# Main execution
if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Generate cave tunnels for MuJoCo")
    parser.add_argument("--count", type=int, default=3, help="Number of caves to generate")
    parser.add_argument("--output-dir", type=str, default="/Users/sam/Documents/MasterThesis/code/Reachbot_RL_Mission/CaveMission/cave_environment/caves", 
                        help="Base output directory for caves")
    args = parser.parse_args()
    
    # Create base caves directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate multiple caves
    caves_summary = []
    for i in range(args.count):
        cave_id = f"cave_{i:03d}"
        print(f"\nGenerating {cave_id}...")
        
        # Create subdirectory for this cave
        cave_dir = os.path.join(args.output_dir, cave_id)
        os.makedirs(cave_dir, exist_ok=True)
        
        # Generate caves with different curve intensities
        curve_intensity = np.random.uniform(0.3, 0.8)  # Random curve intensity for variety
        
        # Generate cave and save files
        cave_params = create_cave(cave_id, cave_dir, curve_intensity)
        caves_summary.append({
            "id": cave_id,
            "directory": cave_dir,
            "vertex_count": cave_params["original_mesh"]["vertex_count"],
            "box_count": cave_params["mujoco_boxes"]["merged_box_count"]
        })
    
    # Save summary of all caves
    summary_path = os.path.join(args.output_dir, "caves_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(caves_summary, f, indent=4)
    
    print(f"\nGenerated {args.count} caves. Summary saved to {summary_path}")


