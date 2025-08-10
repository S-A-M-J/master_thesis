
# open detailed logs json file
import json
import os 

def open_detailed_logs_json_file(file_path):
    """
    Opens a detailed logs JSON file and returns its content.
    
    Args:
        file_path (str): The path to the JSON file.
        
    Returns:
        dict: The content of the JSON file as a dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data

# Get deepest_lidar_direction from the detailed logs for every step
def get_deepest_lidar_directions_from_detailed_logs(file_path):
    """
    Extracts deepest LIDAR directions from a detailed logs JSON file for every step.
    
    Args:
        file_path (str): The path to the detailed logs JSON file.
        
    Returns:
        list: A list of dictionaries containing step info and deepest_lidar_direction.
    """
    data = open_detailed_logs_json_file(file_path)
    
    if not isinstance(data, list):
        raise ValueError("Expected data to be a list of step entries.")
    
    deepest_lidar_directions = []
    
    for step_data in data:
        if 'info' not in step_data:
            raise KeyError(f"'info' field not found in step data: {step_data}")
        
        if 'deepest_lidar_direction' not in step_data['info']:
            raise KeyError(f"'deepest_lidar_direction' not found in info for step {step_data.get('step', 'unknown')}")
        
        deepest_lidar_directions.append({
            'episode': step_data.get('episode', None),
            'step': step_data.get('step', None),
            'deepest_lidar_direction': step_data['info']['deepest_lidar_direction']
        })
    
    return deepest_lidar_directions

# Create new file with deepest_lidar_directions for all steps
def create_deepest_lidar_directions_file(file_path, output_file):
    """
    Creates a new file with deepest LIDAR directions extracted from the detailed logs for every step.
    
    Args:
        file_path (str): The path to the detailed logs JSON file.
        output_file (str): The path to the output file where deepest LIDAR directions will be saved.
    """
    deepest_lidar_directions = get_deepest_lidar_directions_from_detailed_logs(file_path)
    
    with open(output_file, 'w') as file:
        json.dump(deepest_lidar_directions, file, indent=4)
    
    print(f"Deepest LIDAR directions for all steps saved to {output_file}")
    print(f"Total steps processed: {len(deepest_lidar_directions)}")


# Example usage
if __name__ == "__main__":
    detailed_logs_file = '/home/ga53voq/master_thesis/tasks/cave_exploration/logs/cave_exploration-2025-07-27_19-52-36/detailed_logs_episode_0_cave_271_reward_15.68.json'  # Replace with your file path
    output_file = detailed_logs_file.replace('.json', '_deepest_lidar_directions.json')  # Replace with your desired output file path

    try:
        create_deepest_lidar_directions_file(detailed_logs_file, output_file)
    except Exception as e:
        print(f"An error occurred: {e}")