from time import time
import mujoco
import mujoco.viewer
import numpy as np  
from mujoco.glfw import glfw
from BOP_actions import *



def sleep_for(viewer, model, data, duration):
    start_time = time()
    while time() - start_time < duration:
        mujoco.mj_step(model, data)
        viewer.sync()

Object_name = "T"  # "ScrewDriver" "Pentagon"  "T"

if Object_name == "Hexagon":
    model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_Hexagon.xml")
elif Object_name == "Pentagon":
    model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_Pentagon.xml")
elif Object_name == "ScrewDriver":
    model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_ScrewDriver.xml")
elif Object_name == "T":
    model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_T.xml")
elif Object_name == "T2":
    model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_T2.xml")
data = mujoco.MjData(model)


bop_gripper = BOPGripper(model, data, None)


mujoco.mj_resetData(model, data)

# set gravity
model.opt.gravity[2] = 0.0





execution=False
execution_finished = False
with mujoco.viewer.launch_passive(model, data) as viewer:

    bop_gripper.viewer = viewer
    # viewer.cam.lookat[:] = [0.0, 0.4, 0.3]   # center of interest
    # viewer.cam.distance  = .3               # zoom out (bigger = farther)
    # viewer.cam.azimuth   = 270                # rotate around Z
    # viewer.cam.elevation = -40                # tilt down slightly

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")
    while viewer.is_running():

        if data.time > 4.0 and not execution:
            bop_gripper.open_gripper()
            bop_gripper.hand_go_to(pose_name="grasp", Object_name=Object_name)
            bop_gripper.close_gripper()
            bop_gripper.hand_go_to(pose_name="home", Object_name=Object_name) 
            start_time = data.time
            bop_gripper.convey_object(0.05)
            bop_gripper.pitch_object1(rotation=90)   # Morgan wpuld have failed - due to collision
            
            end_time = data.time
            print("Time taken for execution: ", end_time - start_time)

            

            
            # sleep_for(viewer, model, data, duration=1.0)


            execution=True
            execution_finished = True

        if not execution_finished:
            mujoco.mj_step(model, data)
        viewer.sync()
