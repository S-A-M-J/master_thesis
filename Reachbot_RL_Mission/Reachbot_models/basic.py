import os
import numpy as np

class ReachbotConfig:
    """
    Basic model class for Reachbot that provides essential information and mappings.
    """
    
    def __init__(self, num_booms=4, deflection=False):
        """
        Initialize the Reachbot model.
        
        Args:
            xml_path (str): Path to the XML file defining the robot model
            num_booms (int): Number of booms/cables in the model
            action_size (int): Size of the action space, defaults to 2*num_booms if None
        """
        self.xml_path = xml_path if xml_path is not None else os.path.join(
            os.path.dirname(__file__), 'assets', 'reachbot_basic.xml')
        self.num_booms = num_booms
        self.action_size = action_size if action_size is not None else 2 * num_booms
        
    def get_xml_path(self):
        """Return the path to the XML model file."""
        return self.xml_path
    
    def get_num_booms(self):
        """Return the number of booms in the model."""
        return self.num_booms
    
    def get_action_size(self):
        """Return the size of the action space."""
        return self.action_size
    
    def qpos_to_motor_controls(self, qpos):
        """
        Map from qpos state to motor control values.
        
        Args:
            qpos (numpy.ndarray): The joint positions from MuJoCo
            
        Returns:
            numpy.ndarray: Motor control values
        """
        # This is a placeholder implementation - customize based on your specific model
        # Typically this would convert joint positions to motor commands
        motor_controls = np.zeros(self.action_size)
        # Example mapping (to be customized):
        # The first self.num_booms values might control boom extension
        # The next self.num_booms values might control boom orientation
        for i in range(min(len(qpos), self.action_size)):
            motor_controls[i] = qpos[i]  # Simple 1:1 mapping as placeholder
        return motor_controls
    
    def motor_controls_to_qpos(self, motor_controls):
        """
        Map from motor control values to qpos state.
        
        Args:
            motor_controls (numpy.ndarray): Motor control values
            
        Returns:
            numpy.ndarray: The joint positions for MuJoCo
        """
        # This is a placeholder implementation - customize based on your specific model
        # Typically this would convert motor commands to expected joint positions
        qpos = np.zeros(self.action_size)  # Assuming qpos size equals action_size for simplicity
        # Example mapping (to be customized):
        for i in range(min(len(motor_controls), self.action_size)):
            qpos[i] = motor_controls[i]  # Simple 1:1 mapping as placeholder
        return qpos