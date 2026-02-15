from xml.parsers.expat import model
import mujoco
import mujoco.viewer
import numpy as np  
from mujoco.glfw import glfw
import ipdb
import quaternion
import numpy as np

def quat_mul(q1, q2):
    # MuJoCo convention: (w, x, y, z)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
def quat_angle_error(q1, q2):
    q1 = np.array([q1.w,q1.x,q1.y,q1.z ])
    q2 = np.array([q2.w,q2.x,q2.y,q2.z ])

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    return 2 * np.arccos(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))

def quat_to_x_deg(q):
    # q = (w, x, y, z)
    w, x, y, z = q

    # roll (x-axis)
    sinr = 2 * (w*x + y*z)
    cosr = 1 - 2 * (x*x + y*y)
    roll = np.arctan2(sinr, cosr)

    return np.rad2deg(roll)

def quaternion_dist(q1, q2):
    # both must be normalized
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # handle q and -q equivalence
    return 1.0 - abs(np.dot(q1, q2))


def pitch_quat(deg):
    theta = np.deg2rad(deg)
    return np.array([
        np.cos(theta / 2),  # w  # x  ← rotation about X
        0.0,       
        np.sin(theta / 2),         # y
        0.0                 # z
    ])


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




class BOPGripper():
    def __init__(self, model, data, viewer) -> None:
        self.model = model
        self.data = data
        self.viewer = viewer
        self.palm_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.left_belt1_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_belt_1_actuator")
        self.left_belt2_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_belt_2_actuator")
        self.right_belt1_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_belt_1_actuator")
        self.right_belt2_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_belt_2_actuator")
        self.hand_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hand_actuator")

        self.left_belt_1_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "left_belt_1_sensor")
        self.left_belt_2_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "left_belt_2_sensor")
        self.right_belt_1_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_belt_1_sensor")
        self.right_belt_2_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_belt_2_sensor")
        self.palm_pos_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "palm_pos_sensor")
        self.palm_force_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "palm_force_sensor")
        self.hand_pos_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_pos_sensor")
        self.obj_pos_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "obj_pos")
        self.obj_quat_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "obj_quat")

        # ipdb.set_trace()



    def hand_go_to(self, pose_name="home", Object_name=None):
        if pose_name=="home":
            if Object_name == "ScrewDriver":
                position = 0.0
            else:
                position = 0.0
        elif pose_name=="grasp":
            if Object_name == "T":
                position = -0.07 
            elif Object_name == "T2":
                position = -0.09
            else:
                position = -0.123
        

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
        step = 0.01
        # palm pos is 0 to 0.04 and control values are 0 to 255
        palm_pos = self.data.sensordata[self.palm_pos_sensor] * 6375
    
        while palm_pos <245:  
            
            palm_pos += step
            self.data.ctrl[self.palm_act_id] = palm_pos 
            obj_pos = self.data.sensordata[self.obj_pos_sensor]
            obj_quat = self.data.sensordata[self.obj_quat_sensor]

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

    def close_gripper(self):
        step = 0.01
        max_force = 0.1
        palm_pos =  self.data.sensordata[self.palm_pos_sensor] * 6375
        palm_force = self.data.sensordata[self.palm_force_sensor]
        while palm_pos > 0: 

            if palm_force >= max_force :
                # print("Max force reached on both fingers. Stopping closure.")
                return True
            palm_pos -= step
            # print(f"Closing gripper, current palm position: {palm_pos}, palm force: {palm_force}")
            self.data.ctrl[self.palm_act_id] = palm_pos

    
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

    def convey_object(self, position=0.03):
        step = 0.000002
        left_belt1_pos = self.data.sensordata[self.left_belt_1_sensor] 
        left_belt2_pos = self.data.sensordata[self.left_belt_2_sensor] 
        right_belt1_pos = self.data.sensordata[self.right_belt_1_sensor] 
        right_belt2_pos = self.data.sensordata[self.right_belt_2_sensor] 

        object_pos = self.data.sensordata[self.obj_pos_sensor+2]
        init = object_pos
        object_target_pos = position + object_pos
        if position < 0:
            step = -step
  
        while abs(object_pos - object_target_pos) > 0.008 :  
            # print(f"Conveying down object, left_belt1 pos: {left_belt1_pos}, left_belt2 pos: {left_belt2_pos}")
            left_belt1_pos += step
            left_belt2_pos += step
            right_belt1_pos += step
            right_belt2_pos += step
            self.data.ctrl[self.left_belt1_act_id] = left_belt1_pos 
            self.data.ctrl[self.left_belt2_act_id] = left_belt2_pos
            self.data.ctrl[self.right_belt1_act_id] = right_belt1_pos
            self.data.ctrl[self.right_belt2_act_id] = right_belt2_pos

            object_pos = self.data.sensordata[self.obj_pos_sensor+2]

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

    def pitch_object1(self, rotation=30):
        """
        rotation (deg):
        +ve → tip away from palm
        -ve → tip towards palm
        """

        # --- current orientation ---
        current_quaternion = np.array([
            self.data.sensordata[self.obj_quat_sensor + i]
            for i in range(4)
        ])
        current_quaternion /= np.linalg.norm(current_quaternion)

        current_degrees = quat_to_x_deg(current_quaternion)
        target_degrees = current_degrees + rotation


        # --- belt step ---
        step = 0.000006
        if rotation < 0:
            step = -step

        left_belt1_pos  = self.data.sensordata[self.left_belt_1_sensor]
        left_belt2_pos  = self.data.sensordata[self.left_belt_2_sensor]
        right_belt1_pos = self.data.sensordata[self.right_belt_1_sensor]
        right_belt2_pos = self.data.sensordata[self.right_belt_2_sensor]

        # --- closed-loop control in DEGREE space ---
        count=0
        while abs(current_degrees - target_degrees) > 5.0:

            left_belt1_pos  += step
            left_belt2_pos  -= step
            right_belt1_pos += step
            right_belt2_pos -= step

            self.data.ctrl[self.left_belt1_act_id]  = left_belt1_pos
            self.data.ctrl[self.left_belt2_act_id]  = left_belt2_pos
            self.data.ctrl[self.right_belt1_act_id] = right_belt1_pos
            self.data.ctrl[self.right_belt2_act_id] = right_belt2_pos

            mujoco.mj_step(self.model, self.data)

            current_quaternion = np.array([
                self.data.sensordata[self.obj_quat_sensor + i]
                for i in range(4)
            ])
            current_quaternion /= np.linalg.norm(current_quaternion)

            current_degrees = quat_to_x_deg(current_quaternion)
            count+=1
            if count%100==0:
                print(f"[UPDATE] current: {current_degrees:.2f}°, target: {target_degrees:.2f}°, error: ", end="")
                print(f"{abs(current_degrees - target_degrees):.2f}°")
            # if count == 50000:
            #     print("Breaking out of roll loop after 5000 iterations")
            #     break
            self.viewer.sync()

        print(f"[DONE] final: {current_degrees:.2f}°")


    def pitch_object(self, rotation=30):
        """
        rotation (deg):
        +ve → tip away from palm
        -ve → tip towards palm
        """

        # --- current orientation ---
        rotation = np.deg2rad(rotation)
        current_quaternion = quaternion.quaternion(self.data.sensordata[self.obj_quat_sensor ],
                                                   self.data.sensordata[self.obj_quat_sensor +1],
                                                   self.data.sensordata[self.obj_quat_sensor +2],
                                                   self.data.sensordata[self.obj_quat_sensor +3])

        current_transform_mat = quaternion.as_rotation_matrix(current_quaternion)

        # rotate current transform about Y axis by 'rotation' radians
        axis = [0, 1, 0]
        q_rotation = quaternion.from_rotation_vector(rotation * np.array(axis))
        target_quat =  q_rotation * current_quaternion
        


        init = current_quaternion

        # --- belt step ---
        step = 0.000006
        if rotation < 0:
            step = -step

        left_belt1_pos  = self.data.sensordata[self.left_belt_1_sensor]
        left_belt2_pos  = self.data.sensordata[self.left_belt_2_sensor]
        right_belt1_pos = self.data.sensordata[self.right_belt_1_sensor]
        right_belt2_pos = self.data.sensordata[self.right_belt_2_sensor]

        # --- closed-loop control in DEGREE space ---

        while np.rad2deg(quat_angle_error(current_quaternion, target_quat)) >10:

            left_belt1_pos  += step
            left_belt2_pos  -= step
            right_belt1_pos += step
            right_belt2_pos -= step

            self.data.ctrl[self.left_belt1_act_id]  = left_belt1_pos
            self.data.ctrl[self.left_belt2_act_id]  = left_belt2_pos
            self.data.ctrl[self.right_belt1_act_id] = right_belt1_pos
            self.data.ctrl[self.right_belt2_act_id] = right_belt2_pos

            mujoco.mj_step(self.model, self.data)

            current_quaternion = quaternion.quaternion(self.data.sensordata[self.obj_quat_sensor ],
                                                   self.data.sensordata[self.obj_quat_sensor +1],
                                                   self.data.sensordata[self.obj_quat_sensor +2],
                                                   self.data.sensordata[self.obj_quat_sensor +3])
            

            self.viewer.sync()

        print("current", current_quaternion, "target ", target_quat)

    def roll_object(self, rotation=30):
        """
        rotation (deg):
        +ve → tip away from palm
        -ve → tip towards palm
        """

        # --- current orientation ---
        rotation = np.deg2rad(rotation)
        current_quaternion = quaternion.quaternion(self.data.sensordata[self.obj_quat_sensor ],
                                                   self.data.sensordata[self.obj_quat_sensor +1],
                                                   self.data.sensordata[self.obj_quat_sensor +2],
                                                   self.data.sensordata[self.obj_quat_sensor +3])

        axis = [1, 0, 0]
        q_rotation = quaternion.from_rotation_vector(rotation * np.array(axis))

        target_quat =  q_rotation * current_quaternion
        init = current_quaternion

        # --- belt step ---
        step = 0.000006
        if rotation < 0:
            step = -step

        left_belt1_pos  = self.data.sensordata[self.left_belt_1_sensor]
        left_belt2_pos  = self.data.sensordata[self.left_belt_2_sensor]
        right_belt1_pos = self.data.sensordata[self.right_belt_1_sensor]
        right_belt2_pos = self.data.sensordata[self.right_belt_2_sensor]

        # --- closed-loop control in DEGREE space ---
        count=0
        while np.rad2deg(quat_angle_error(current_quaternion, target_quat)) >11:

            left_belt1_pos  += step
            left_belt2_pos  += step
            right_belt1_pos -= step
            right_belt2_pos -= step

            self.data.ctrl[self.left_belt1_act_id]  = left_belt1_pos
            self.data.ctrl[self.left_belt2_act_id]  = left_belt2_pos
            self.data.ctrl[self.right_belt1_act_id] = right_belt1_pos
            self.data.ctrl[self.right_belt2_act_id] = right_belt2_pos

            mujoco.mj_step(self.model, self.data)

            current_quaternion = quaternion.quaternion(self.data.sensordata[self.obj_quat_sensor ],
                                                   self.data.sensordata[self.obj_quat_sensor +1],
                                                   self.data.sensordata[self.obj_quat_sensor +2],
                                                   self.data.sensordata[self.obj_quat_sensor +3])
            
            count+=1
            # if count%100==0:
            #     print("initial angle error:",np.rad2deg(quat_angle_error(current_quaternion, target_quat)))
            #     print("changed angle:",np.rad2deg(quat_angle_error(init, current_quaternion)))
            
            self.viewer.sync()


    def roll_object_clockwise(self, position=0.03): 
        step = 0.00001
        left_belt1_pos = self.data.sensordata[self.left_belt_1_sensor] 
        left_belt2_pos = self.data.sensordata[self.left_belt_2_sensor] 
        right_belt1_pos = self.data.sensordata[self.right_belt_1_sensor] 
        right_belt2_pos = self.data.sensordata[self.right_belt_2_sensor] 
        left_belt1_target_pos = left_belt1_pos + position

        while abs(left_belt1_pos - left_belt1_target_pos) > 0.001 :  
            left_belt1_pos += step
            left_belt2_pos += step
            right_belt1_pos -= step
            right_belt2_pos -= step
            self.data.ctrl[self.left_belt1_act_id] = left_belt1_pos 
            self.data.ctrl[self.left_belt2_act_id] = left_belt2_pos
            self.data.ctrl[self.right_belt1_act_id] = right_belt1_pos
            self.data.ctrl[self.right_belt2_act_id] = right_belt2_pos

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
