import mujoco
import mujoco.viewer
import numpy as np  
from mujoco.glfw import glfw



class ActuatorTorque:
    def __init__(self) -> None:
        self.dyn = np.array([1, 0, 0])
        self.gain = np.array([1, 0, 0])
        self.bias = np.array([0, 0, 0])

    def __repr__(self) -> str:
        return f"ActuatorTorque(dyn={self.dyn}, gain={self.gain}, bias={self.bias})"


class ActuatorPosition:
    def __init__(self) -> None:
        self.dyn = np.array([1, 0, 0])
        self.gain = np.array([21.1, 0, 0])
        self.bias = np.array([0, -21.1, 0])

    def __repr__(self) -> str:
        return f"ActuatorPosition(dyn={self.dyn}, gain={self.gain}, bias={self.bias})"





def update_actuator(model, actuator_id, actuator):
    """
    Update actuator in model
    model - mujoco.MjModel
    actuator_id - int or str (name) (for reference see, named access to model elements)
    actuator - ActuatorTorque, ActuatorPosition, ActuatorVelocity
    """

    model.actuator(actuator_id).dynprm[:3] = actuator.dyn
    model.actuator(actuator_id).gainprm[:3] = actuator.gain
    model.actuator(actuator_id).biasprm[:3] = actuator.bias




class VFGripper():
    def __init__(self, model, data, viewer) -> None:
        self.model = model
        self.data = data
        self.viewer = viewer
        self.finger_act_id_1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger1")
        self.friction_act_id_1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger1_friction")
        self.finger_act_id_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger2")
        self.friction_act_id_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger2_friction")
        self.palm_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_palm")
        self.palm_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger1_slide")
        self.finger1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger1_to_motor1")
        self.finger2_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger2_to_motor2")

        self.finger_sensor_pos_id_1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "finger1_pos")
        self.finger_sensor_pos_id_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "finger2_pos")
        self.finger_sensor_torque_id_1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "finger1_force")
        self.finger_sensor_torque_id_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "finger2_force")
        self.hand_pos_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_pos_sensor")
        
        self.hand_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hand_actuator")
        
        self.set_torque = 0.15



    def hand_go_to(self, pose_name="home", Object_name=None):
        if pose_name=="home":
            if Object_name == "ScrewDriver":
                position = -0.12
            else:
                position = 0.0
        elif pose_name=="grasp":
            if Object_name == "ScrewDriver":
                position = -0.123 -0.12
            else:
                position = -0.123 - 0.06
        

        hand_pos = self.data.sensordata[self.hand_pos_sensor] 
        if hand_pos < position:
            step = 0.00001
        else:
            step = -0.00001
        while abs(hand_pos - position) > 0.001: 
            
            hand_pos += step
            self.data.ctrl[self.hand_actuator_id] = hand_pos    
            # print(f"Moving hand to {pose_name} pose, current hand pos: {hand_pos}") 


            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

    def open_gripper(self):
        step = 0.0001
        posl =  self.data.sensordata[0]
        pos2 =  self.data.sensordata[3]
        while posl > -0.4 or pos2 > -0.4:   
            
            posl -= step
            pos2 -= step
            if posl > -0.4 :  
                self.data.ctrl[self.finger_act_id_1] = posl

            if pos2 > -0.4 :  
                self.data.ctrl[self.finger_act_id_2] = pos2


            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

    def close_gripper(self):
        step = 0.0001
        max_force = 0.1
        posl =  self.data.sensordata[self.finger_sensor_pos_id_1]
        pos2 =  self.data.sensordata[self.finger_sensor_pos_id_2]
        force1 = self.data.sensordata[self.finger_sensor_torque_id_1]
        force2 = self.data.sensordata[self.finger_sensor_torque_id_2]
        while posl < 0.3 and pos2 < 0.3:   

            if force1 >= max_force or force2 >= max_force:
                # print("Max force reached on both fingers. Stopping closure.")
                return True 
            posl += step
            pos2 += step
            
            if pos2 < 0.3 :  
                self.data.ctrl[self.finger_act_id_2] = pos2

            if posl < 0.3 :  
                self.data.ctrl[self.finger_act_id_1] = posl


            mujoco.mj_step(self.model, self.data)
            force1 = self.data.sensordata[2]
            force2 = self.data.sensordata[5]
            self.viewer.sync()


    def set_palm_width(self, width_mm):
        step = 0.00001
        width_m = width_mm / 1000.0
        target_pos = -0.0402 + (width_m / 2.0) 
        # print(f"Setting palm width to {width_mm} mm, target actuator pos: {target_pos}")
        pos = self.data.joint(self.palm_joint_id).qpos
        if pos < target_pos:
            step = step
        else:
            step = -step
        while abs(pos - target_pos) > 0.001:
            pos += step
            self.data.ctrl[self.palm_act_id] = pos
            # print(f"Current palm actuator pos: {pos}, target: {target_pos}")

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
        return True


    def adapt_palm_width_to_object(self):
        pos1 =  self.data.joint(self.finger1_joint_id).qpos
        self.set_finger_torque(0.1, self.finger_act_id_1)
        self.set_finger_torque(0.1, self.finger_act_id_2)

        error = pos1   # becasue 0 is the center position
        if error>0:
            sign_i=1
            sign_j=-1
        else:
            sign_i=-1
            sign_j=1
        # print(f"Adapting palm width to object, initial error: {error}")
        count=0
        #check for sign change
        while abs(error) > 1e-7 :
            palm_pos = self.data.joint(self.palm_joint_id).qpos
            palm_pos -= error * 0.0000005
            self.data.ctrl[self.palm_act_id] = palm_pos

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

            pos1 =  self.data.joint(self.finger1_joint_id).qpos
            error = pos1 
            if error<0:
                sign_j=1
            else:
                sign_j=-1
            if sign_i == sign_j:
                print(f"Adapting palm width to object, current error: {error}")
                # print("Sign change detected, stopping adaptation.")
                break
            count+=1

            if count%100==0:
                print(f"pos1: {pos1}")
                print(f"Adjusting palm position to: {palm_pos}")
                print(f"Adapting palm width to object, current error: {error}")
                pass
        
        self.set_finger_torque(0.1, self.finger_act_id_1)
        self.set_finger_torque(0.1, self.finger_act_id_2)

        return True
    
    def move_finger(self, finger_act_id, finger_sensor_id, position):
        step = 0.0001
        curr_pos = self.data.sensordata[finger_sensor_id]
        while abs(curr_pos - position) > 0.001:
            if curr_pos < position:
                curr_pos += step   
            else:
                curr_pos -= step

            # print(f"Moving finger to position: {curr_pos} and target position: {position}")
            self.data.ctrl[finger_act_id] = curr_pos
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
        return True

    def move_finger_with_palm(self, finger_act_id, finger_sensor_id, position):
        step = 0.00001
        pos1 =  self.data.joint(self.finger1_joint_id).qpos
        pos2 =  self.data.joint(self.finger2_joint_id).qpos
        curr_pos = self.data.sensordata[finger_sensor_id]
        error = pos1 + pos2
        count=0
        while abs(curr_pos - position) > 0.001:
            if curr_pos < position:
                curr_pos += step   
            else:
                curr_pos -= step

            # print(f"Moving finger to position: {curr_pos} and target position: {position}")
            self.data.ctrl[finger_act_id] = curr_pos
            palm_pos = self.data.joint(self.palm_joint_id).qpos
            palm_pos -= error * 0.005
            if count%500==0:
                self.data.ctrl[self.palm_act_id] = palm_pos
                pass

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            pos1 =  self.data.joint(self.finger1_joint_id).qpos
            pos2 =  self.data.joint(self.finger2_joint_id).qpos
            error = pos1 + pos2
            count+=1

            if count%100==0:
                print(f"pos1: {pos1}, pos2: {pos2}")
                print(f"palm position to: {palm_pos}")

                print(f"palm width to object, current error: {error}")
        

        return True


    def set_finger_torque(self, torque, finger_act_id):
        update_actuator(self.model, finger_act_id, ActuatorTorque())
        self.data.ctrl[finger_act_id] = torque
        return True
    
    def set_finger_position(self, finger_act_id):
        update_actuator(self.model, finger_act_id, ActuatorPosition())
        pos = self.data.sensordata[finger_act_id]
        self.data.ctrl[finger_act_id] = pos
        return True


    def set_friction_state(self, actuator_id, friction_state):
        actuator_state = -0.001 if friction_state == "High" else 0.0038
        self.data.ctrl[actuator_id] = actuator_state
        return True


    

    def slide_on_right_finger_dn(self, position=0.3):
        self.set_friction_state(self.friction_act_id_1, friction_state="High")
        self.set_friction_state(self.friction_act_id_2, friction_state="Low")
        self.set_finger_position(self.finger_act_id_1)
        self.set_finger_torque(self.set_torque, self.finger_act_id_2)
        self.move_finger(self.finger_act_id_1, self.finger_sensor_pos_id_1, position=position)
        return True

    def slide_on_right_finger_up(self, position=0.3):

        self.set_friction_state(self.friction_act_id_2, friction_state="Low")
        self.set_friction_state(self.friction_act_id_1, friction_state="High")
        self.set_finger_position(self.finger_act_id_2)
        self.set_finger_torque(self.set_torque, self.finger_act_id_1)
        self.move_finger(self.finger_act_id_2, self.finger_sensor_pos_id_2, position=position)
        return True


    def slide_on_left_finger_up(self, position=0.3):
        self.set_friction_state(self.friction_act_id_1, friction_state="Low")
        self.set_friction_state(self.friction_act_id_2, friction_state="High")
        self.set_finger_position(self.finger_act_id_1)
        self.set_finger_torque(self.set_torque, self.finger_act_id_2)
        self.move_finger(self.finger_act_id_1, self.finger_sensor_pos_id_1, position=position)
        return True

    def slide_on_left_finger_dn(self, position=0.3):

        self.set_friction_state(self.friction_act_id_1, friction_state="Low")
        self.set_friction_state(self.friction_act_id_2, friction_state="High")
        self.set_finger_position(self.finger_act_id_2)
        self.set_finger_torque(self.set_torque, self.finger_act_id_1)
        self.move_finger(self.finger_act_id_2, self.finger_sensor_pos_id_2, position=position)
        return True

    def rotate_object_counter_clockwise(self, position=0.7):
        self.set_friction_state(self.friction_act_id_1, friction_state="High")
        self.set_friction_state(self.friction_act_id_2, friction_state="High")
        self.set_finger_position(self.finger_act_id_1)
        self.set_finger_torque(0.1, self.finger_act_id_2)
        self.move_finger(self.finger_act_id_1, self.finger_sensor_pos_id_1, position=position)
 

    def rotate_object_clockwise(self, position=0.3):
        self.set_friction_state(self.friction_act_id_1, friction_state="High")
        self.set_friction_state(self.friction_act_id_2, friction_state="High")
        self.set_finger_position(self.finger_act_id_2)
        self.set_finger_torque(0.1, self.finger_act_id_1)
        self.move_finger(self.finger_act_id_2, self.finger_sensor_pos_id_2, position=position)

            