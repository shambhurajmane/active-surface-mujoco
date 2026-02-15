import trimesh
import numpy as np
from scripts.Planner_common_files.class_definitions import State, Cost
import quaternion
from scripts.Planner_common_files.geometry_fcl_utils import normalize_quaternion
import ipdb
def orient_prism(mesh, obj_ori):
    w, x, y, z = obj_ori.w, obj_ori.x, obj_ori.y, obj_ori.z
    quat = [w, x, y, z]
    matrix = trimesh.scene.transforms.kwargs_to_matrix(quaternion=quat)
    mesh.apply_transform(matrix)
    return mesh


def switching_action_cost(prev_action, action):
    # Cost of switching fingers
    if prev_action != action:
        return Cost.SwitchingCost 
    else:
        return 0  


def check_start_goal_state_validity(state, collision_checker, plan_config, obj_mesh):
    new_mesh = orient_prism(obj_mesh.copy(), state.obj_ori)
    smallest = state.left_f if state.left_f < state.right_f else state.right_f
    new_vert,_= trimesh.remesh.subdivide_to_size(new_mesh.vertices, new_mesh.faces,1)
    rounding_factor = 2 
    total_vertices = np.round(np.array(new_vert), rounding_factor) # round the vertices to 2 decimal places
    palm_width = collision_checker.calculate_palm_width(total_vertices, smallest, state)
    if palm_width is None:
        return False   # object is dropped, invalid state
    valid, left_most_x = collision_checker.check_convey_up(total_vertices, palm_width, state)
    if valid and collision_checker.is_valid_bop_state(state, left_most_x, plan_config):
        return True
    else:
        return False

def get_neighbors(current_state ,mesh, collsion_checker=None, plan_config=None):
    ##############################################   Get geometric data     ################################################
    new_mesh = orient_prism(mesh.copy(), current_state.obj_ori)
    smallest = current_state.left_f if current_state.left_f < current_state.right_f else current_state.right_f
    new_vert,_= trimesh.remesh.subdivide_to_size(new_mesh.vertices,new_mesh.faces,1)
    rounding_factor = 2 
    total_vertices = np.round(np.array(new_vert), rounding_factor) # round the vertices to 2 decimal places
    palm_width = collsion_checker.calculate_palm_width(total_vertices, smallest, current_state)
    # print("Palm width:", palm_width)
    # print("curent")
    # current_state.show_state()


    if palm_width is None:
        return []   # object is dropped, no valid neighbors

    

    ##############################################   Get neighbors    ################################################
    neighbors = []
    debug = plan_config["debug"]             
    if debug:
        print("Generating neighbors for state:")
        current_state.show_state()

    slide_resolution = plan_config["slide_resolution"]
    roll_resolution = np.deg2rad(plan_config["roll_resolution"])
    pitch_resolution = np.deg2rad(plan_config["pitch_resolution"])



    # neighbor1 : Convey up on both fingers
    r_finger, l_finger = current_state.right_f  + slide_resolution, current_state.left_f + slide_resolution
    switching_cost = switching_action_cost(current_state.prev_action, "Translate_on_both_fingers_Up")
    new_state = State(l_finger, r_finger, current_state.dist_z, current_state.obj_ori, current_state, current_state.g_cost + Cost.convey + switching_cost, palm_width)   
    valid, left_most_x = collsion_checker.check_convey_up(total_vertices, palm_width, new_state)
    if valid and collsion_checker.is_valid_bop_state(new_state, left_most_x, plan_config):
        new_state.prev_action = "Translate_on_both_fingers_Up"
        neighbors.append(new_state)
        if debug :
            print("Translate_on_both_fingers_Up")
            new_state.show_state()

    # neighbor1 : Convey down on both fingers
    r_finger, l_finger = current_state.right_f  - slide_resolution, current_state.left_f - slide_resolution
    switching_cost = switching_action_cost(current_state.prev_action, "Translate_on_both_fingers_Down")
    new_state = State(l_finger, r_finger, current_state.dist_z, current_state.obj_ori, current_state, current_state.g_cost + Cost.convey + switching_cost, palm_width)   
    valid, left_most_x = collsion_checker.check_convey_down(total_vertices, palm_width, new_state)
    if valid and collsion_checker.is_valid_bop_state(new_state, left_most_x, plan_config):
        new_state.prev_action = "Translate_on_both_fingers_Down"
        neighbors.append(new_state)
        if debug :
            print("Translate_on_both_fingers_Down")
            new_state.show_state()


    # # Roll object by angle_step along z-axis 
    axis = [0, 0, 1]
    q_rotation = quaternion.from_rotation_vector(roll_resolution * np.array(axis))
    # Apply the rotation
    new_orientation = q_rotation * current_state.obj_ori
    new_orientation = normalize_quaternion(new_orientation)
    r_finger, l_finger = current_state.right_f , current_state.left_f 
    switching_cost = switching_action_cost(current_state.prev_action, "Roll_obj_clockwise")  
    new_state = State(l_finger, r_finger, current_state.dist_z, new_orientation, current_state, current_state.g_cost + Cost.roll+ switching_cost, None) 
    valid, left_most_x = collsion_checker.check_convey_up(total_vertices, palm_width, new_state)
    if valid and collsion_checker.is_valid_bop_state(new_state, left_most_x, plan_config):
        new_state.prev_action = "Roll_obj_clockwise"
        neighbors.append(new_state)

        if debug :
            print("Roll_obj_clockwise")
            new_state.show_state()


    axis = [0, 0, 1]
    q_rotation = quaternion.from_rotation_vector(-roll_resolution * np.array(axis))
    # Apply the rotation
    new_orientation = q_rotation * current_state.obj_ori
    new_orientation = normalize_quaternion(new_orientation)
    r_finger, l_finger = current_state.right_f , current_state.left_f 
    switching_cost = switching_action_cost(current_state.prev_action, "Roll_obj_counterclockwise")  
    new_state = State(l_finger, r_finger, current_state.dist_z, new_orientation, current_state, current_state.g_cost + Cost.roll+ switching_cost, None) 
    valid, left_most_x = collsion_checker.check_convey_up(total_vertices, palm_width, new_state)
    if valid and collsion_checker.is_valid_bop_state(new_state, left_most_x, plan_config):
        new_state.prev_action = "Roll_obj_counterclockwise"
        neighbors.append(new_state)

        if debug :
            print("Roll_obj_counterclockwise")
            new_state.show_state()


    
    # Pitch object by angle_step along y-axis
    axis = [0, 1, 0]
    q_rotation = quaternion.from_rotation_vector(pitch_resolution * np.array(axis))
    # Apply the rotation
    new_orientation = q_rotation * current_state.obj_ori
    new_orientation = normalize_quaternion(new_orientation)
    r_finger, l_finger = current_state.right_f , current_state.left_f 
    switching_cost = switching_action_cost(current_state.prev_action, "Pitch_obj_clockwise") 
    
    # dist_z_movement = np.round(current_state.dist_z * np.cos(rotation_resolution), decimals=2)  # update dist_z due to pitch 
    # along_finger_movement = np.round(current_state.dist_z * np.sin(rotation_resolution), decimals=2)
    dist_z_movement = 0
    along_finger_movement = 0
    new_state = State(l_finger + along_finger_movement, r_finger + along_finger_movement, dist_z_movement, new_orientation, current_state, current_state.g_cost + Cost.pitch+ switching_cost, None) 
    valid, left_most_x = collsion_checker.check_convey_up(total_vertices, palm_width, new_state)
    if valid and collsion_checker.is_valid_bop_state(new_state, left_most_x, plan_config):
        new_state.prev_action = "Pitch_obj_clockwise"
        neighbors.append(new_state)

        if debug :
            print("Pitch_obj_clockwise")
            new_state.show_state()


    axis = [0, 1, 0]
    q_rotation = quaternion.from_rotation_vector(-pitch_resolution * np.array(axis))
    # Apply the rotation
    new_orientation = q_rotation * current_state.obj_ori
    new_orientation = normalize_quaternion(new_orientation)
    r_finger, l_finger = current_state.right_f , current_state.left_f 
    switching_cost = switching_action_cost(current_state.prev_action, "Pitch_obj_counterclockwise")  
    # dist_z_movement = np.round(current_state.dist_z * np.cos(-rotation_resolution), decimals=2)  # update dist_z due to pitch 
    # along_finger_movement = np.round(current_state.dist_z * np.sin(-rotation_resolution), decimals=2)
    dist_z_movement = 0
    along_finger_movement = 0
    new_state = State(l_finger + along_finger_movement, r_finger + along_finger_movement, dist_z_movement, new_orientation, current_state, current_state.g_cost + Cost.pitch+ switching_cost, None) 
# valid, leftmost_pt_x = collsion_checker.check_rot(total_vertices, new_state, palm_width)
    valid, left_most_x = collsion_checker.check_convey_up(total_vertices, palm_width, new_state)
    if valid and collsion_checker.is_valid_bop_state(new_state, left_most_x, plan_config):
        new_state.prev_action = "Pitch_obj_counterclockwise"
        neighbors.append(new_state)

        if debug :
            print("Pitch_obj_counterclockwise")
            new_state.show_state()
    
    # ipdb.set_trace()
    return neighbors    
    
