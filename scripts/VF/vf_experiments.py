import mujoco
import mujoco.viewer
import numpy as np  
from mujoco.glfw import glfw
from VF_actions import *


Object_name = "Hexagon"  # "ScrewDriver" "Pentagon"  "T"

if Object_name == "Hexagon":
    model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\VF_test.xml")
elif Object_name == "Pentagon":
    model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_Pentagon.xml")
elif Object_name == "ScrewDriver":
    model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_ScrewDriver.xml")
elif Object_name == "T":
    model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_T.xml")
data = mujoco.MjData(model)


vf_gripper = VFGripper(model, data, None)


mujoco.mj_resetData(model, data)

# set gravity
model.opt.gravity[2] = 0.0


execution=False
with mujoco.viewer.launch_passive(model, data) as viewer:

    vf_gripper.viewer = viewer
    # viewer.cam.lookat[:] = [0.0, 0.4, 0.3]   # center of interest
    # viewer.cam.distance  = .3               # zoom out (bigger = farther)
    # viewer.cam.azimuth   = 270                # rotate around Z
    # viewer.cam.elevation = -40                # tilt down slightly

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")
    while viewer.is_running():

        if data.time > 4.0 and not execution:
            vf_gripper.open_gripper()
            # vf_gripper.hand_go_to(pose_name="grasp", Object_name=Object_name)
            vf_gripper.set_palm_width(20)  #10mm
            vf_gripper.close_gripper()
            vf_gripper.adapt_palm_width_to_object()
            print("Adapted palm width to object")
            # vf_gripper.hand_go_to(pose_name="home", Object_name=Object_name)
            # vf_gripper.slide_on_right_finger_up(0.4)
            # vf_gripper.slide_on_left_finger_up(0)
            vf_gripper.set_friction_state(vf_gripper.friction_act_id_1, "High")

            vf_gripper.rotate_object_counter_clockwise(0.6)
            vf_gripper.slide_on_right_finger_up(0)
            # vf_gripper.slide_on_right_finger_dn(0.4)
            # vf_gripper.rotate_object_clockwise(0.6)    
            # vf_gripper.slide_on_right_finger_dn(0)

            execution=True
            print("Set palm width to 40mm")

        mujoco.mj_step(model, data)
        viewer.sync()
