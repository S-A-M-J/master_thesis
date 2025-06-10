
from mujoco_playground._src import mjx_env
from reachbot.joystick import Joystick
from reachbot.getup import Getup
from reachbot.reachbot_constants import ROOT_PATH

def get_reachbot_joystick_env()-> mjx_env.MjxEnv:
    reachbot_env = Joystick()
    #reachbot_env = Joystick(str(xml_path), config)
    return reachbot_env

def get_reachbot_getup_env()-> mjx_env.MjxEnv:
    return Getup()

