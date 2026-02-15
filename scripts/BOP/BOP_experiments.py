from time import time
import mujoco
import mujoco.viewer
import numpy as np  
from mujoco.glfw import glfw
from scripts.BOP.BOP_actions import *


def sleep_for(viewer, model, data, duration):
    start_time = time()
    while time() - start_time < duration:
        mujoco.mj_step(model, data)
        viewer.sync()




def run_bop_experiment(action_list=None, Object_name="T", configs=None):


    if Object_name == "Hexagon_BOP":
        model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_Hexagon.xml")
    elif Object_name == "Pentagon_BOP":
        model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_Pentagon.xml")
    elif Object_name == "Screwdriver_BOP":
        model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_ScrewDriver.xml")
    elif Object_name == "T_BOP1":
        model = mujoco.MjModel.from_xml_path(r".\mujoco\test_cases\BOP_T.xml")
    elif Object_name == "BOP_T_2":
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

                for action_DICT in action_list:
                    #ex {'Translate_on_both_fingers_Up': 3}
                    for action, Value in action_DICT.items():

                        print("Executing action: ", action)
                        print("Value: ", Value)
                        if action == "Translate_on_both_fingers_Up":
                            slide_step = configs['bop_planner_params']['slide_resolution'] / 100.0
                            bop_gripper.convey_object(slide_step*Value)
                        elif action == "Translate_on_both_fingers_Down":
                            slide_step = configs['bop_planner_params']['slide_resolution'] / 100.0
                            bop_gripper.convey_object(-slide_step)
                        elif action == "Roll_obj_clockwise":
                            roll_step = configs['bop_planner_params']['roll_resolution']
                            bop_gripper.roll_object(rotation=roll_step*Value)
                        elif action == "Roll_obj_counterclockwise":
                            roll_step = configs['bop_planner_params']['roll_resolution']
                            bop_gripper.roll_object(rotation=-roll_step*Value)
                        elif action == "Pitch_obj_clockwise":
                            pitch_step = configs['bop_planner_params']['pitch_resolution']
                            bop_gripper.pitch_object1(rotation=pitch_step*Value)  
                        elif action == "Pitch_obj_counterclockwise":
                            pitch_step = configs['bop_planner_params']['pitch_resolution']
                            bop_gripper.pitch_object1(rotation=-pitch_step*Value) 
                
                end_time = data.time
                print("Time taken for execution: ", end_time - start_time)

                

                
                # sleep_for(viewer, model, data, duration=1.0)


                execution=True
                execution_finished = True

            if not execution_finished:
                mujoco.mj_step(model, data)
            viewer.sync()



if __name__ == "__main__":
    run_bop_experiment()