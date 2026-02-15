import numpy as np
from scripts.Planner_common_files.utils import get_config_from_yaml

configs = get_config_from_yaml()


class State:
    def __init__(self,left_f, right_f, dist_z, obj_ori, parent=None , g_cost=0, obj_width=None):
        self.left_f = left_f    
        self.right_f = right_f
        self.dist_z = dist_z
        self.obj_ori = obj_ori
        self.parent = parent
        self.g_cost = g_cost                # execution cost
        self.h_cost = 0                     # heuristic cost
        self.prev_action = "None"
        self.obj_width = obj_width
        self.state_index = self.find_state_index()

    def show_state(self):
        print(f"Left finger: {self.left_f}, Right finger: {self.right_f}, Dist Z: {self.dist_z}, Obj Ori: {self.obj_ori}, G cost: {self.g_cost}, H cost: {self.h_cost}, Total cost: {self.g_cost + self.h_cost}")
   

    def __lt__(self, other):
        return (self.g_cost + self.h_cost) < (other.g_cost + other.h_cost)
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        if (np.isclose(self.left_f, other.left_f, atol=1e-3) and
            np.isclose(self.right_f, other.right_f, atol=1e-3) and
            np.isclose(self.dist_z, other.dist_z, atol=1e-3)):
            
            if configs["experiments_params"]["gripper_name"] == "vf_hand":
                rotation_error_margin = configs['vf_planner_params']['rotation_error_margin']
            elif configs["experiments_params"]["gripper_name"] == "bop":
                rotation_error_margin = configs['bop_planner_params']['rotation_error_margin']
            return (np.allclose(self.obj_ori, other.obj_ori, atol=rotation_error_margin) or
                    np.allclose(self.obj_ori, -other.obj_ori, atol=rotation_error_margin))
        return False

    def find_state_index(self, step=0.01, slide_resolution=1):
        slide_resolution = configs['vf_planner_params']['slide_resolution']
        state_ori = np.array([self.obj_ori.w, self.obj_ori.x, self.obj_ori.y, self.obj_ori.z])
        if state_ori[0] < 0:
            state_ori = -state_ori  # enforce quaternion sign consistency

        ori_tuple = tuple(np.round(state_ori / step) * step)
        return (
            round(self.left_f / slide_resolution) * slide_resolution,
            round(self.right_f / slide_resolution) * slide_resolution,
            round(self.dist_z / slide_resolution) * slide_resolution,
            ori_tuple
        )

class Cost:             # Cost of each action to account for in A* algorithm : Shambhuraj
    if configs["experiments_params"]["gripper_name"] == "vf_hand":
        planner_costs =configs['vf_planner_params']['planner_costs']
        R_cw = planner_costs['R_cw']
        R_ccw = planner_costs['R_ccw']
        S_UP = planner_costs['S_UP'] * configs['vf_planner_params']['slide_resolution']
        S_DN = planner_costs['S_DN'] * configs['vf_planner_params']['slide_resolution']
        MOV_UP = planner_costs['MOV_UP']
        MOV_DN = planner_costs['MOV_DN']
        Pivot = planner_costs['Pivot']
        SwitchingCost = planner_costs['SwitchingCost']
    elif configs["experiments_params"]["gripper_name"] == "bop":
        planner_costs =configs['bop_planner_params']['planner_costs']
        convey = planner_costs['convey']
        roll = planner_costs['roll']
        pitch = planner_costs['pitch']
        SwitchingCost = planner_costs['SwitchingCost']

class VF():
    y_bounds = [-5, 5]    # maximum palm width ie object can only be grasped in these bounds
    x_bounds = [2, 12]      # finger surface width
    z_bounds = [-1.5, 1.5]     # finger surface length
    finger_width = 3        # finger width  

    z_planner_bounds = [0, 11] # planner bounds
