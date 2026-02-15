from scripts.Planner_common_files.utils import get_config_from_yaml
from scripts.Planner_common_files.case_definition import get_case
from scripts.Planner_common_files.geometry_fcl_utils import CollsionCheck
import numpy as np
import time

import quaternion
from heapdict import heapdict
from scripts.Planner_common_files.class_definitions import State
resolution = None 
debug = None
configs = get_config_from_yaml()
from scripts.Planner_common_files.utils import save_data



def heuristic_quaternion_distance(state, goal_state):
    # heuristic function is the quaternion distance between the current orientation and the goal orientation
    state_ori = state.obj_ori
    goal_ori = goal_state.obj_ori
    #inner product of two quaternions
    inner_product = state_ori.w*goal_ori.w + state_ori.x*goal_ori.x + state_ori.y*goal_ori.y + state_ori.z*goal_ori.z
    quaternion_distance = np.arccos(2 * inner_product**2 - 1)
    return quaternion_distance



def heuristic_eucl(state, goal_state):
    # heuristic function is the eucledian distance between the current location and the goal location (left_f, right_f, dist_z)
    return np.sqrt((state.left_f - goal_state.left_f)**2 + (state.right_f - goal_state.right_f)**2 + (state.dist_z - goal_state.dist_z)**2)
    
def heuristic_l1_norm(state, goal_state):
    # heuristic function is the l1 norm distance between the current location and the goal location (left_f, right_f, dist_z)
    return abs(state.left_f - goal_state.left_f) + abs(state.right_f - goal_state.right_f) + abs(state.dist_z - goal_state.dist_z)



 
def process_actions(path):  
    '''
    Docstring for process_actions
    
    :param path: List of State objects representing the planned path
    :return:
    action_sequence - List of actions in the planned sequence, ex : [{"action_name": "R_Cw", "obj_width": 5, "start_state": "x y z", "goal_state": "x y z"}, ...]
    action_list - List of actions with counts of consecutive actions, ex : [{"R_Cw": 3}, {"Pivot": 1}, ...]
    no_of_switches - Number of times the action switched
    pivot_count - Number of pivot actions in the path
    rotation_count - Number of rotation actions in the path

    ''' 
    action_sequence = []
    action_list = []
    no_of_switches = 0  
    pivot_count = 0
    rotation_count = 0  
    count = 0
    for i in range(len(path)-1):
        action_data = {}
        current = path[i]
        next = path[i+1]
        action = next.prev_action
        if action == "Pivot":   
            pivot_count += 1
        if action == "R_Cw" or action == "R_Ccw":
            rotation_count += 1

        if action != current.prev_action:   
            no_of_switches += 1 
            action_list.append({action: 1})
            count = 1
        else:
            count += 1
            action_list.pop()
            action_list.append({action: count})
        action_data["action_name"] = action
        action_data["obj_width"] = next.obj_width
        action_data["start_state"] = str(current.left_f) + " " + str(current.right_f) + " " + str(current.dist_z)
        action_data["start_ori"] = str(current.obj_ori) 
        action_data["goal_state"] = str(next.left_f) + " " + str(next.right_f) + " " + str(next.dist_z)
        action_data["goal_ori"] = str(next.obj_ori) 

        action_sequence.append(action_data)

    print("Action List with counts of consecutive actions: ")
    print(action_list)
    
    # print(action_sequence)   
    return action_sequence, action_list ,no_of_switches, pivot_count, rotation_count   


def a_star(start_state, goal_state, mesh, plan_config, collsion_checker): 
    '''
    Docstring for a_star
    
    :param start_state: State object representing the start state
    :param goal_state: State object representing the goal state
    :param mesh: Mesh of the object to be manipulated loaded using trimesh
    :param plan_config: Configuration parameters for planning
    :param collsion_checker: CollsionCheck object for collision checking
    :return:
    path - List of State objects representing the planned path
    success - Boolean indicating if the planning was successful
    '''


    closedSet = {}
    debug_state = []
    costQueue = heapdict()
    openSet = {start_state.state_index: start_state} # Initialize the open set with the start node
    costQueue[start_state.state_index] = start_state.g_cost   # Add start mode into priority queue
    orientation_list =[]
    exploration_dict = {}
    heuristic_weights = plan_config["heuristic_weights"]
    m_factor = float(plan_config["m_factor"])

    while not openSet == {}:
        
        current_node_index = costQueue.popitem()[0]  # Get the node with the lowest cost 
        current_state = openSet.pop(current_node_index)  # Get the state of the node with the lowest cost
        
        closedSet[current_node_index] = current_state  # Add the current node to the closed set 
        debug_state.append(current_state)   

        if current_state == goal_state:   # Goal test
            print("number of nodes explored: ", len(debug_state) )
            path = [current_state]
            current_state.show_state()
            while current_state.parent:
                path.append(current_state.parent)
                current_state = current_state.parent
                current_state.show_state()
            print("path length: ", len(path))       
            return path[::-1] , True, debug_state, exploration_dict  # Return the path in reverse order
    
        
        for neighbor in get_neighbors(current_state, mesh, collsion_checker, plan_config):
            neighbor_index = neighbor.state_index
            heuristic_cost = 0
            if float(heuristic_weights[0]) > 0:
                heuristic_cost += float(heuristic_weights[0]) * heuristic_quaternion_distance(neighbor, goal_state)
            elif float(heuristic_weights[1]) > 0:
                heuristic_cost += float(heuristic_weights[1]) * heuristic_eucl(neighbor, goal_state)    
            elif float(heuristic_weights[2]) > 0:
                heuristic_cost += float(heuristic_weights[2]) * heuristic_l1_norm(neighbor, goal_state)
      
            if neighbor_index not in closedSet.keys():    
                if neighbor_index not in openSet :
                    openSet[neighbor_index] = neighbor
                    costQueue[neighbor_index] = neighbor.g_cost + heuristic_cost * m_factor
                elif neighbor.g_cost < openSet[neighbor_index].g_cost:   
                    openSet[neighbor_index] = neighbor
                    costQueue[neighbor_index] = neighbor.g_cost + heuristic_cost * m_factor

    print("Failed to find a path")
    print("number of nodes explored: ", len(debug_state) )          
    return orientation_list, False, debug_state, exploration_dict

def plot_explored_states(debug_state):
    import matplotlib.pyplot as plt
    dist_zs = [x.dist_z for x in debug_state]  
    left_fs = [x.left_f for x in debug_state]
    right_fs = [x.right_f for x in debug_state]
    obj_oris = [x.obj_ori for x in debug_state]
    prev_actions = [x.prev_action for x in debug_state]
    total_pivots = prev_actions.count("Pivot")
    unique_dist_zs = list(set(dist_zs)) 
    counts = [dist_zs.count(x) for x in unique_dist_zs]
    plt.figure()
    plt.bar(unique_dist_zs, counts)
    plt.xlabel("dist_z values")
    plt.ylabel("Number of states explored")
    plt.title("Explored states distribution over dist_z values")
    plt.show()

def planner(object_name, id, configs):         
    '''
    Docstring for planner
    
    :param object_name: Object selecteed for planning
    :param id: Case ID for planning
    :param configs: Configuration parameters loaded from YAML file
    :return: 
    action_sequence - List of actions in the planned sequence, ex : [{"action_name": "R_Cw", "obj_width": 5, "start_state": "x y z", "goal_state": "x y z"}, ...]
    action_list - List of actions with counts of consecutive actions, ex : [{"R_Cw": 3}, {"Pivot": 1}, ...]
    no_of_switches - Number of times the action switched
    pivot_count - Number of pivot actions in the path
    rotation_count  - Number of rotation actions in the path
    time_taken - Time taken for planning in seconds
    states_explored - Number of states explored during planning

    '''

    global resolution, debug
    if configs["experiments_params"]["gripper_name"] == "vf_hand":
        plan_config = configs['vf_planner_params']
    elif configs["experiments_params"]["gripper_name"] == "bop":
        plan_config = configs['bop_planner_params']
    debug = plan_config["debug"]
    resolution = plan_config["resolution"]
    
    # Get the case data
    obj_mesh, finger_mesh, start_state, goal_state, object_data = get_case(object_name, id)

    collsion_checker = CollsionCheck(finger_mesh)

    start_state = State(start_state["left_f"], start_state["right_f"], start_state["dist_z"], start_state["obj_ori"], obj_width=start_state["obj_width"])
    
    goal_state = State(goal_state["left_f"],goal_state["right_f"], goal_state["dist_z"], goal_state["obj_ori"], obj_width=goal_state["obj_width"])

    if debug:
        start_state.show_state()
        goal_state.show_state()
        collsion_checker.visualize_scene(start_state, start_state.obj_width, obj_mesh.copy())
        collsion_checker.visualize_scene(goal_state, goal_state.obj_width, obj_mesh.copy()) 
        if configs["experiments_params"]["gripper_name"] == "vf_hand":
            print(collsion_checker.is_valid_state(start_state,0))
            print(collsion_checker.is_valid_state(goal_state,0))
        elif configs["experiments_params"]["gripper_name"] == "bop":
            print(check_start_goal_state_validity(start_state, collsion_checker, plan_config, obj_mesh.copy()))
            print(check_start_goal_state_validity(goal_state, collsion_checker, plan_config, obj_mesh.copy()))


    start_time = time.time()
    path, success, debug_state, exploration_dict = a_star(start_state, goal_state, obj_mesh, plan_config, collsion_checker) 
    time_taken = time.time() - start_time  
    print("Object name : ", object_name, " Case ID : ", id)
    print("Time taken for planning (s): ", time_taken)

    
    if success:
        action_sequence, action_list ,no_of_switches, pivot_count, rotation_count    = process_actions(path)
    else:
        plot_explored_states(debug_state)
        action_sequence, action_list ,no_of_switches, pivot_count, rotation_count = [], [], 0, 0, 0

    return action_sequence, action_list ,no_of_switches, pivot_count, rotation_count , time_taken, len(debug_state)



if __name__ == "__main__":
    gripper_name = configs["experiments_params"]["gripper_name"]
    if gripper_name == "vf_hand":
        from scripts.VF.vf_neighbors import get_neighbors

    elif gripper_name == "bop":
        from scripts.BOP.bop_neighbors import get_neighbors, check_start_goal_state_validity
        from scripts.BOP.BOP_experiments import run_bop_experiment
    object_name = configs["experiments_params"]["object_name"]
    id = configs["experiments_params"]["case_number"]
    action_sequence, action_list ,no_of_switches, pivot_count, rotation_count,time_taken, states_explored = planner(object_name, id, configs)
    print("Planned Action Sequence: ")
    print(action_sequence)
    if gripper_name == "bop":
        run_bop_experiment(action_list, object_name, configs)

    # for id in range(1,17):
    #     action_sequence, action_list ,no_of_switches, pivot_count, rotation_count,time_taken, states_explored = planner(object_name, id, configs)
    #     save_data(object_name, id, action_sequence, action_list ,no_of_switches, pivot_count, rotation_count, time_taken, states_explored)
    #     print("-----------------------------------------------------")
    # for object_name in ["PentagonBOP2", "PentagonBOP3", "PentagonBOP4"]:
    #     action_sequence, action_list ,no_of_switches, pivot_count, rotation_count,time_taken, states_explored = planner(object_name, id, configs)
    #     save_data(object_name, id, action_sequence, action_list ,no_of_switches, pivot_count, rotation_count, time_taken, states_explored)
