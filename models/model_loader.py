from enum import Enum
import mujoco
from mujoco import MjModel
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__))


class ReachbotModelType(Enum):
    BASIC = "basic"
    DEFLECTION = "deflection"

class ReachbotModel:
    def __init__(self, model_type: ReachbotModelType):
        self.model_type = model_type
        if model_type == ReachbotModelType.BASIC:
            self._model_path = MODEL_DIR + "/basic/reachbot.xml"
        elif model_type == ReachbotModelType.DEFLECTION:
            self._model_path = MODEL_DIR + "/deflection/reachbot.xml"

    
    def render_model(self):
        """
        Renders the model using the MuJoCo viewer.
        """
        # Load model and create data
        print(f"Loading model from: {self._model_path}")
        model = mujoco.MjModel.from_xml_path(self._model_path)
        data = mujoco.MjData(model)

        mujoco.mj_resetDataKeyframe(model, data,  0)

        # Launch viewer
        with mujoco.viewer.launch(model, data) as viewer:
            # Initial camera setup
            viewer.cam.lookat[:] = [0, 0, 0]  # Looking at the center
            viewer.cam.distance = 20.0  # Adjusted distance
            viewer.cam.azimuth = 122.8  # Based on xyaxes rotation
            viewer.cam.elevation = -40.3  # Based on xyaxes elevation
            # Position approximately matches pos="-12.870 -7.169 14.348"
            # Orientation approximately matches xyaxes="0.542 -0.841 0.000 0.532 0.343 0.774"

            # Render loop
            while viewer.is_running():
                viewer.sync()
                mujoco.mj_step(model, data)

    # Accesors

    @property
    def model_path(self):
        return self._model_path