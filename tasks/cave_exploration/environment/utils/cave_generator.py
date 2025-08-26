import numpy as np
import trimesh
import os
import json
import argparse
import time
from trimesh.voxel import creation
from tqdm import tqdm

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

def generate_cave_tunnel(curve_intensity=0.5, seed=None):
    # Set random seed for reproducible generation
    if seed is not None:
        np.random.seed(seed)
    
    # Parameters
    length = 20.0  # meters
    radius = 1.5  # meters
    num_segments = 10
    points_per_circle = 12
    offset_magnitude = 2
    num_dents = 3  # Number of dents to add
    
    # Randomly choose segments for dents (excluding first and last)
    available_segments = list(range(3, num_segments - 1))  # All segments except first 3 and last
    dented_segments = np.random.choice(
        available_segments, 
        size=min(num_dents, len(available_segments)), 
        replace=False
    )
    
    #print(f"Adding dents to segments: {dented_segments}")
    
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
    #print(f"DEBUG: centerline[0] = {centerline[0]}")
    #print(f"DEBUG: centerline[-1] = {centerline[-1]}")
    
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
    #     #print(f"Warning: Could not repair mesh: {e}")
    #     #print("Continuing with voxelization of unrepaired mesh.")
    
    try:
        # Voxelize the mesh
        voxel_grid = mesh.voxelized(pitch=voxel_size)
        
        # Get filled voxels
        voxels = voxel_grid.sparse_indices
        
        # Return both the voxel grid and points
        return voxel_grid, voxels
    except Exception as e:
        #print(f"Error during voxelization: {e}")
        #print("Returning empty voxel grid")
        # Return empty data if voxelization fails
        return None, np.array([])

def save_cave_config(voxels, voxel_size, filename, cave_params, cave_id=None):
    """Save complete cave configuration with voxel positions, voxel size, and all cave information"""
    # Start with the existing cave parameters and add voxel information
    cave_config = cave_params.copy()
    
    # Add voxel-specific information
    cave_config["voxel_size"] = voxel_size
    cave_config["box_count"] = len(voxels)
    cave_config["boxes"] = []
    
    # Add each voxel as a box to the configuration
    for i, voxel in enumerate(voxels):
        # Convert voxel coordinates to world coordinates (center of the voxel)
        x = float(voxel[0] * voxel_size + voxel_size / 2)
        y = float(voxel[1] * voxel_size + voxel_size / 2)
        z = float(voxel[2] * voxel_size + voxel_size / 2)
        
        # Size is half the voxel size in each dimension (MuJoCo convention)
        half_voxel_size = float(voxel_size / 2)
        
        box_data = {
            "id": i,
            "position": {
                "x": x,
                "y": y,
                "z": z
            },
        }
        cave_config["boxes"].append(box_data)
    
    # Update box stats in the config
    cave_config["cave_boxes"] = {
        "box_count": len(voxels)
    }
    
    # Save configuration to JSON file
    with open(filename, 'w') as f:
        json.dump(cave_config, f, indent=4)
    
    return len(voxels)

def create_cave(cave_id, output_dir, curve_intensity=0.5, voxel_size=0.2, seed=None):
    """Create a single cave and save to the specified directory"""
    # Set random seed for reproducible generation
    if seed is not None:
        np.random.seed(seed)
    
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
        "curve_intensity": curve_intensity,
        "voxel_size": voxel_size,
        "seed": seed
    }
    
    # Generate the original mesh with the parameters
    cave_mesh = generate_cave_tunnel(curve_intensity, seed)
    #print(f"Cave {cave_id} - Original mesh - Vertex count: {len(cave_mesh.vertices)}")
    #print(f"Cave {cave_id} - Original mesh - Face count: {len(cave_mesh.faces)}")
    
    # Save mesh stats
    cave_params["original_mesh"] = {
        "vertex_count": len(cave_mesh.vertices),
        "face_count": len(cave_mesh.faces),
        "is_watertight": cave_mesh.is_watertight
    }
    
    try:
        # Save original mesh
        mesh_path = os.path.join(output_dir, "cave_tunnel_original.stl")
        #print(f"Saving original mesh to {mesh_path}")
        cave_mesh.export(mesh_path)
    except Exception as e:
        print(f"Error saving original mesh: {e}")
    
    # Voxelize the mesh with configurable resolution
    voxel_grid, voxels = voxelize_mesh(cave_mesh, voxel_size)
    #print(f"Cave {cave_id} - Voxelized mesh - Number of voxels: {len(voxels)}")
    
    # Save voxelization stats
    cave_params["voxelization"] = {
        "voxel_size": voxel_size,
        "voxel_count": len(voxels)
    }
    
    # Calculate and save voxel bounds if voxels exist
    if len(voxels) > 0:
        # Convert voxel indices to world coordinates early for bounds calculation
        world_voxels = voxels * voxel_size
        
        # Calculate bounds
        voxel_bounds = {
            "x_min": float(np.min(world_voxels[:, 0])),
            "x_max": float(np.max(world_voxels[:, 0])),
            "y_min": float(np.min(world_voxels[:, 1])),
            "y_max": float(np.max(world_voxels[:, 1])),
            "z_min": float(np.min(world_voxels[:, 2])),
            "z_max": float(np.max(world_voxels[:, 2]))
        }
        
        cave_params["voxel_bounds"] = voxel_bounds
        #print(f"Cave {cave_id} - Voxel bounds: x=[{voxel_bounds['x_min']:.2f}, {voxel_bounds['x_max']:.2f}], "
        #      f"y=[{voxel_bounds['y_min']:.2f}, {voxel_bounds['y_max']:.2f}], "
        #      f"z=[{voxel_bounds['z_min']:.2f}, {voxel_bounds['z_max']:.2f}]")
    else:
        cave_params["voxel_bounds"] = {
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "z_min": 0.0,
            "z_max": 0.0
        }
        #print(f"Cave {cave_id} - No voxels found, setting bounds to zero")

    # Calculate target position based on voxel data
    if len(voxels) > 0:

        start_mix_x = np.min(voxels[:, 0])
        start_voxels = voxels[voxels[:, 0] == start_mix_x]
        start_min_y = np.min(start_voxels[:, 1])
        start_max_y = np.max(start_voxels[:, 1])
        start_mid_y = (start_max_y + start_min_y) / 2
        start_min_z = np.min(start_voxels[:, 2])
        start_z = start_min_z + 0.2  # 0.2m
        # Add 5 voxel segments before the start position to flat start ground
        for i in range(5):
            for j in range(len(start_voxels)):
                voxels = np.vstack((voxels, start_voxels[j] + np.array([-i - 1, 0, 0])))

        # Close cave entrance by filling gaps between lowest and highest z for each y row
        # Find the segment with the smallest x (after adding the 5 segments)
        min_x_after_extension = np.min(voxels[:, 0])
        entrance_voxels = voxels[voxels[:, 0] == min_x_after_extension]
        
        # Get unique y values and sort them
        unique_y = np.unique(entrance_voxels[:, 1])
        
        # For each y row, fill gaps between min and max z if there are more than 2 voxels
        entrance_fill_voxels = []
        for y_val in unique_y:
            y_row_voxels = entrance_voxels[entrance_voxels[:, 1] == y_val]
            
            if len(y_row_voxels) > 1:  # Only fill if more than 2 voxels in the row
                z_min = np.min(y_row_voxels[:, 2])
                z_max = np.max(y_row_voxels[:, 2])
                
                # Fill all z values between min and max
                for z_val in range(int(z_min) + 1, int(z_max)):
                    new_voxel = np.array([min_x_after_extension, y_val, z_val])
                    entrance_fill_voxels.append(new_voxel)
        
        # Add the new voxels to close the entrance
        if entrance_fill_voxels:
            entrance_fill_voxels = np.array(entrance_fill_voxels)
            voxels = np.vstack((voxels, entrance_fill_voxels))
            #print(f"Cave {cave_id} - Added {len(entrance_fill_voxels)} voxels to close cave entrance")

        # Shift all voxels to have 0 position at middle of min_x
        voxels[:, 1] -= int(round(start_mid_y))
        voxels[:, 2] -= int(round(start_z))

        # Update world coordinates after shifting
        world_voxels = voxels * voxel_size

        # Update voxel bounds after shifting
        cave_params["voxel_bounds"] = {
            "x_min": float(np.min(world_voxels[:, 0])),
            "x_max": float(np.max(world_voxels[:, 0])),
            "y_min": float(np.min(world_voxels[:, 1])),
            "y_max": float(np.max(world_voxels[:, 1])),
            "z_min": float(np.min(world_voxels[:, 2])),
            "z_max": float(np.max(world_voxels[:, 2]))
        }
        #print(f"Cave {cave_id} - Updated voxel bounds after shifting: x=[{cave_params['voxel_bounds']['x_min']:.2f}, {cave_params['voxel_bounds']['x_max']:.2f}], "
        #      f"y=[{cave_params['voxel_bounds']['y_min']:.2f}, {cave_params['voxel_bounds']['y_max']:.2f}], "
        #      f"z=[{cave_params['voxel_bounds']['z_min']:.2f}, {cave_params['voxel_bounds']['z_max']:.2f}]")
        
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
        
        #print(f"Cave {cave_id} - Target position: x={max_x:.2f}, y={mid_y:.2f}, z={target_z:.2f}")
        
        # Calculate starting positions for x steps 0 to 15
        starting_positions = []
        # Number of segments in the cave x direction
        num_cave_segments = int(np.ceil(length / voxel_size))
        for x_step in range(num_cave_segments - 5):  # 0 to num_cave_segments
            # Convert x step to actual x coordinate in voxel space
            x_coord = x_step
            
            # Find all voxels at this x coordinate
            x_voxels = voxels[voxels[:, 0] == x_coord]
            
            if len(x_voxels) > 0:
                # Find y range at this x
                y_min = np.min(x_voxels[:, 1])
                y_max = np.max(x_voxels[:, 1])
                y_mid = (y_max + y_min) / 2
                
                # Find voxels at the middle y position (or closest to it)
                closest_y = x_voxels[np.argmin(np.abs(x_voxels[:, 1] - y_mid)), 1]
                mid_y_voxels = x_voxels[x_voxels[:, 1] == closest_y]
                
                # Find lowest z among these voxels
                z_min = np.min(mid_y_voxels[:, 2])
                
                # Starting position is 2 voxel sizes above the lowest z
                start_pos = {
                    "x": float(x_coord * voxel_size),
                    "y": float(closest_y * voxel_size),
                    "z": float((z_min + 2) * voxel_size)
                }
                starting_positions.append(start_pos)
        
        # Save starting positions
        cave_params["starting_pos"] = starting_positions
        #print(f"Cave {cave_id} - Generated {len([p for p in starting_positions if p is not None])} starting positions")
        
    
    # Convert the final voxels (with added segments and entrance filled) to a new mesh
    if len(voxels) > 0:
        # Create voxel boxes directly as a mesh using trimesh primitives
        from trimesh import creation
        
        # Create individual box meshes for each voxel and combine them
        box_meshes = []
        for voxel in voxels:
            # Convert voxel coordinates to world coordinates
            box_center = voxel * voxel_size
            # Create a box mesh at this position
            box_mesh = creation.box(extents=[voxel_size, voxel_size, voxel_size])
            # Translate to the correct position
            box_mesh.apply_translation(box_center)
            box_meshes.append(box_mesh)
        
        # Combine all box meshes into one
        if box_meshes:
            voxel_mesh = trimesh.util.concatenate(box_meshes)
            #print(f"Cave {cave_id} - Voxel mesh (final voxels) - Vertex count: {len(voxel_mesh.vertices)}")
            #print(f"Cave {cave_id} - Voxel mesh (final voxels) - Face count: {len(voxel_mesh.faces)}")
            # Save voxel mesh stats
            cave_params["voxel_mesh"] = {
                "vertex_count": len(voxel_mesh.vertices),
                "face_count": len(voxel_mesh.faces)
            }
            # Save voxelized mesh as STL
            voxel_mesh.export(os.path.join(output_dir, "cave_tunnel_voxels.stl"))
        else:
            print(f"Cave {cave_id} - No voxel boxes created")
            cave_params["voxel_mesh"] = {
                "vertex_count": 0,
                "face_count": 0
            }
    else:
        print(f"Cave {cave_id} - Skipping voxel mesh creation due to voxelization failure")
        cave_params["voxel_mesh"] = {
            "vertex_count": 0,
            "face_count": 0
        }
    
    # Create cave configuration with voxel positions and all cave information
    box_count = save_cave_config(voxels, voxel_size, os.path.join(output_dir, "cave_config.json"), cave_params, cave_id)
    #print(f"Cave {cave_id} - Created {box_count} voxel boxes")
    
    # Save box stats
    cave_params["cave_boxes"] = {
        "box_count": box_count
    }
    
    #print(f"Cave {cave_id} information saved to cave_config.json")
    
    return cave_params

# Main execution
if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Generate cave tunnels for MuJoCo")
    parser.add_argument("--count", type=int, default=3, help="Number of caves to generate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "caves"),
        help="Base output directory for caves (default: ../caves relative to this script)"
    )
    parser.add_argument("--voxel-size", type=float, default=0.2, help="Voxel size in meters (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation (default: 42)")
    args = parser.parse_args()
    
    # Create base caves directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate multiple caves
    caves_summary = []
    
    # Set initial seed for global randomness
    np.random.seed(args.seed)
    
    for i in tqdm(range(args.count), desc="Generating caves", unit="cave"):
        cave_id = i + 1
        
        # Create subdirectory for this cave
        cave_dir = os.path.join(args.output_dir, f"cave_{cave_id:03d}")
        os.makedirs(cave_dir, exist_ok=True)
        
        # Generate deterministic seed for this cave based on base seed and cave id
        cave_seed = args.seed + cave_id * 1000
        
        # Generate caves with different curve intensities using the cave-specific seed
        np.random.seed(cave_seed)
        curve_intensity = np.random.uniform(0.3, 0.8)  # Random curve intensity for variety
        
        # Generate cave and save files
        cave_params = create_cave(cave_id, cave_dir, curve_intensity, args.voxel_size, cave_seed)
        
        caves_summary.append({
            "id": cave_id,
            "directory": cave_dir,
            "vertex_count": cave_params["original_mesh"]["vertex_count"],
            "box_count": cave_params["cave_boxes"]["box_count"],
            "seed": cave_seed,
            "curve_intensity": curve_intensity
        })
    
    # Save summary of all caves
    summary_path = os.path.join(args.output_dir, "caves_summary.json")
    summary_data = {
        "generation_info": {
            "base_seed": args.seed,
            "cave_count": args.count,
            "voxel_size": args.voxel_size,
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "caves": caves_summary
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    
    print(f"\nGenerated {args.count} caves with voxel size {args.voxel_size}m using seed {args.seed}. Summary saved to {summary_path}")


