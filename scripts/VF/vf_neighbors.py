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




def get_neighbors(current_state ,mesh, collsion_checker=None, plan_config=None):
    ##############################################   Get geometric data     ################################################
    new_mesh = orient_prism(mesh.copy(), current_state.obj_ori)
    smallest = current_state.left_f if current_state.left_f < current_state.right_f else current_state.right_f
    new_vert,_= trimesh.remesh.subdivide_to_size(new_mesh.vertices,new_mesh.faces,1)
    rounding_factor = 2 
    total_vertices = np.round(np.array(new_vert), rounding_factor) # round the vertices to 2 decimal places
    palm_width = collsion_checker.calculate_palm_width(total_vertices, smallest, current_state)

    if palm_width is None:
        return []   # object is dropped, no valid neighbors

    ##############################################   Get neighbors    ################################################
    neighbors = []
    debug = False             
    if debug:
        print("Generating neighbors for state:")
        current_state.show_state()

    slide_resolution = plan_config["slide_resolution"]

    # neighbor1 : slide up on left finger
    r_finger, l_finger = current_state.right_f, current_state.left_f + slide_resolution
    switching_cost = switching_action_cost(current_state.prev_action, "Slide_on_Left_Finger_Up")
    new_state = State(l_finger, r_finger, current_state.dist_z, current_state.obj_ori, current_state, current_state.g_cost + Cost.S_UP + switching_cost, palm_width)   
    valid, left_most_x = collsion_checker.check_slide_left_up(total_vertices, palm_width, new_state)
    if valid and collsion_checker.is_valid_state(new_state, left_most_x, plan_config):
        new_state.prev_action = "Slide_on_Left_Finger_Up"
        neighbors.append(new_state)
        if debug :
            print("Slide_on_Left_Finger_Up")
            new_state.show_state()


    # neighbor2 : slide down on left finger
    r_finger, l_finger = current_state.right_f, current_state.left_f - slide_resolution
    switching_cost = switching_action_cost(current_state.prev_action, "Slide_on_Left_Finger_Dn") 
    new_state = State(l_finger, r_finger, current_state.dist_z, current_state.obj_ori, current_state, current_state.g_cost + Cost.S_DN+ switching_cost, palm_width)   
    valid, left_most_x = collsion_checker.check_slide_left_down(total_vertices, palm_width, new_state)
    if valid and collsion_checker.is_valid_state(new_state, left_most_x, plan_config) :
        new_state.prev_action = "Slide_on_Left_Finger_Dn"
        neighbors.append(new_state)
        if debug :
            print("Slide_on_Left_Finger_Dn")
            new_state.show_state()

    # neighbor3 : slide up on right finger
    r_finger, l_finger = current_state.right_f + slide_resolution, current_state.left_f
    switching_cost = switching_action_cost(current_state.prev_action, "Slide_on_Right_Finger_Up")
    new_state = State(l_finger, r_finger, current_state.dist_z, current_state.obj_ori, current_state, current_state.g_cost + Cost.S_UP+ switching_cost, palm_width) 
    valid, left_most_x = collsion_checker.check_slide_right_up(total_vertices, palm_width, new_state)
    if valid and collsion_checker.is_valid_state(new_state, left_most_x, plan_config):
        new_state.prev_action = "Slide_on_Right_Finger_Up"
        neighbors.append(new_state)
        if debug :
            print("Slide_on_Right_Finger_Up")
            new_state.show_state()

    # neighbor4 : slide down on right finger
    r_finger, l_finger = current_state.right_f - slide_resolution, current_state.left_f
    switching_cost = switching_action_cost(current_state.prev_action, "Slide_on_Right_Finger_Dn")
    new_state = State(l_finger, r_finger, current_state.dist_z, current_state.obj_ori, current_state, current_state.g_cost + Cost.S_DN+ switching_cost, palm_width) 
    valid, left_most_x = collsion_checker.check_slide_right_down(total_vertices, palm_width, new_state)
    if valid and collsion_checker.is_valid_state(new_state, left_most_x, plan_config):
        new_state.prev_action = "Slide_on_Right_Finger_Dn"
        neighbors.append(new_state)
        if debug :
            print("Slide_on_Right_Finger_Dn")
            new_state.show_state()

    if current_state.left_f == current_state.right_f:
        # neighbor7 : move contact up
        switching_cost = switching_action_cost(current_state.prev_action, "Move_up")
        new_state = State(current_state.right_f, current_state.left_f, current_state.dist_z+1, current_state.obj_ori, current_state, current_state.g_cost + Cost.MOV_UP+ switching_cost, palm_width)
        if collsion_checker.check_move_contact_up(total_vertices, palm_width, smallest, new_state):
            new_state.prev_action = "Move_up"
            neighbors.append(new_state)
            if debug :
                print("Move_up")
                new_state.show_state()

        # neighbor8 : move contact down
        switching_cost = switching_action_cost(current_state.prev_action, "Move_dn")
        new_state = State(current_state.right_f, current_state.left_f, current_state.dist_z-1, current_state.obj_ori, current_state, current_state.g_cost + Cost.MOV_DN+ switching_cost, palm_width)
        if collsion_checker.check_move_contact_dn(total_vertices, palm_width, smallest, new_state):
            new_state.prev_action = "Move_dn"
            neighbors.append(new_state)
            if debug :
                print("Move_dn")
                new_state.show_state()  


    # # neighbor5 : Rotate anticlockwise
    ca, cca, cw_slide, ccw_slide = collsion_checker.find_data_from_mesh(total_vertices, current_state, palm_width)
    if np.round(ca,1) != 0 and np.rad2deg(ca)>10 and cw_slide != 0:
        axis = [0, 0, 1]
        q_rotation = quaternion.from_rotation_vector(ca * np.array(axis))
        # Apply the rotation
        new_orientation = q_rotation * current_state.obj_ori
        new_orientation = normalize_quaternion(new_orientation)
        r_finger, l_finger = current_state.right_f + cw_slide, current_state.left_f - cw_slide
        switching_cost = switching_action_cost(current_state.prev_action, "Rotate_obj_anticlockwise")  
        new_state = State(l_finger, r_finger, current_state.dist_z, new_orientation, current_state, current_state.g_cost + Cost.R_cw+ switching_cost, None) 
        valid, leftmost_pt_x = collsion_checker.check_rot(total_vertices, new_state, palm_width)
        if valid and collsion_checker.is_valid_state(new_state, leftmost_pt_x, plan_config):
            new_state.prev_action = "Rotate_obj_anticlockwise"
            neighbors.append(new_state)
            if debug :
                print("Rotate_obj_anticlockwise")
                new_state.show_state()


        # # neighbor6 : Rotate clockwise
    if np.round(cca,1) != 0 and np.rad2deg(cca)>10 and ccw_slide != 0:
        axis = [0, 0, 1]
        q_rotation = quaternion.from_rotation_vector(-cca * np.array(axis))
        # Apply the rotation
        new_orientation = q_rotation * current_state.obj_ori
        new_orientation = normalize_quaternion(new_orientation)
        r_finger, l_finger = current_state.right_f - ccw_slide, current_state.left_f + ccw_slide
        switching_cost = switching_action_cost(current_state.prev_action, "Rotate_obj_clockwise")
        new_state = State(l_finger, r_finger, current_state.dist_z, new_orientation, current_state, current_state.g_cost + Cost.R_ccw+ switching_cost, None)  
        valid, leftmost_pt_x = collsion_checker.check_rot(total_vertices, new_state, palm_width)

        if valid and collsion_checker.is_valid_state(new_state, leftmost_pt_x, plan_config):
            new_state.prev_action = "Rotate_obj_clockwise"
            neighbors.append(new_state)
            if debug :
                print("Rotate_obj_clockwise")
                new_state.show_state()



    # # neighbor9 : Pivot Horizontal and vertical
    if current_state.left_f == current_state.right_f:
        r_finger, l_finger = current_state.right_f + current_state.dist_z, current_state.left_f + current_state.dist_z
        switching_cost = switching_action_cost(current_state.prev_action, "Pivot_Horizontal")
        angle_step = np.deg2rad(90)
        axis = [0, 1, 0]
        q_rotation = quaternion.from_rotation_vector(angle_step * np.array(axis))
        # Apply the rotation 
        new_orientation = q_rotation * current_state.obj_ori
        new_orientation = normalize_quaternion(new_orientation)
        valid, pivot_action, new_dist_z, leftmost_pt_x = collsion_checker.check_pivot_horizontal(total_vertices, current_state, palm_width, plan_config)
        new_state = State(l_finger, r_finger, 0, new_orientation, current_state, current_state.g_cost + Cost.Pivot+ switching_cost, palm_width)  
        if valid and collsion_checker.is_valid_state(new_state, leftmost_pt_x, plan_config):
            new_state.prev_action = pivot_action
            new_state.dist_z = 0
            neighbors.append(new_state)
            if debug :
                print("Pivot_Horizontal")

    if current_state.left_f == current_state.right_f:
        r_finger, l_finger = current_state.right_f , current_state.left_f 
        switching_cost = switching_action_cost(current_state.prev_action, "Pivot_Vertical")
        angle_step = np.deg2rad(90)
        axis = [0, 1, 0]
        q_rotation = quaternion.from_rotation_vector(angle_step * np.array(axis))
        # Apply the rotation 
        new_orientation = q_rotation * current_state.obj_ori
        new_orientation = normalize_quaternion(new_orientation)
        valid, pivot_action, new_dist_z, leftmost_pt_x = collsion_checker.check_pivot_horizontal(total_vertices, current_state, palm_width, plan_config)
        new_state = State(l_finger, r_finger, 0, new_orientation, current_state, current_state.g_cost + Cost.Pivot+ switching_cost, palm_width)  
        if valid and collsion_checker.is_valid_state(new_state, leftmost_pt_x, plan_config):
            new_state.prev_action = pivot_action
            new_state.dist_z = new_dist_z
            neighbors.append(new_state)
            if debug :
                print("Pivot_Vertical")
                   

    return neighbors    
    
