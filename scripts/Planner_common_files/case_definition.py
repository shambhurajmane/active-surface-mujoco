import trimesh
import os
import numpy as np
import quaternion
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


def get_mesh(object_name,case_id):   
    path = r".\assets\TestCases"
    simple_objects = ["HexagonBig","HexagonSmall","RectangleBig", "RectangleSmall", "SquareBig", "SquareSmall","Rubik" ] 
    YCB_Objects_Test_Cases = ["cracker_box","gelatin_box","potted_meat_can","spatula","spoon","tomato_soup_can", "gelatin_box_choco"]
    Complex_Test_Cases = ["T","L"]
    Diffusion_objects = ["Cube", "Hexagon", "Star"]
    region_based_objects =  ["square_prism", "hexagonal_prism_large", "rectangular_prism_small", "rectangular_prism_large", "rectangular_prism_curved"  ,"hexagonal_prism_tall", "hexagonal_prism_small"]
    bop_cases = ["Hexagon_BOP", "T_BOP", "Pentagon_BOP", "Screwdriver_BOP", "PentagonBOP2", "PentagonBOP3", "PentagonBOP4"]
    case_name =  "case"+str(case_id)+".stl" 
    if object_name in simple_objects:
        mesh_path = os.path.join(path,"Simple_Objects_Test_Cases",object_name,case_name)
        mesh = trimesh.load(mesh_path)
    elif object_name in YCB_Objects_Test_Cases:
        mesh_path = os.path.join(path,"YCB_Objects_Test_Cases",object_name,"cases", case_name) 
        mesh = trimesh.load(mesh_path)
    elif object_name in Complex_Test_Cases:
        mesh_path = os.path.join(path,"Complex_Test_Cases",object_name,case_name)
        mesh = trimesh.load(mesh_path)
    elif object_name in region_based_objects:
        mesh_path = os.path.join(path,"region_based_objects",object_name,case_name)
        mesh = trimesh.load(mesh_path)
    elif object_name in Diffusion_objects:
        mesh_path = os.path.join(path,"Diffusion_objects",object_name,case_name)
        mesh = trimesh.load(mesh_path)
    elif object_name == "Hexagon_16_cases":
        mesh_path = os.path.join(path,"Hexagon_16_cases",case_name)
        mesh = trimesh.load(mesh_path)
    elif object_name in bop_cases:
        mesh_path = os.path.join(path,"BOP_test_cases",object_name +".stl")
        mesh = trimesh.load(mesh_path)
    else:
        mesh_path = os.path.join(path,"new_generated_test_case","test.stl")
        mesh = trimesh.load(mesh_path)

    return mesh 




def find_orientation(start_ori_meshes, goal_ori_meshes):
    initial_positions = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])       
    for mesh in start_ori_meshes.keys():    
        centroid = start_ori_meshes[mesh] - start_ori_meshes["origin"] 
        
        # print(centroid) 
        if mesh == "s":
            centroid = np.round(centroid / 0.5,2)
            initial_positions[0][0] = centroid[0]
            initial_positions[0][1] = centroid[1]   
            initial_positions[0][2] = centroid[2]   
        elif mesh == "m":
            centroid = np.round(centroid / 0.75,2)
            initial_positions[1][0] = centroid[0]   
            initial_positions[1][1] = centroid[1]
            initial_positions[1][2] = centroid[2] 
        elif mesh == "l":
            initial_positions[2][0] = centroid[0]
            initial_positions[2][1] = centroid[1]
            initial_positions[2][2] = centroid[2]

    final_positions = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]) 
    for mesh in goal_ori_meshes.keys():
        centroid = goal_ori_meshes[mesh] - goal_ori_meshes["origin"] 
        if mesh == "s":
            centroid = np.round(centroid / 0.5,2)
            final_positions[0][0] = centroid[0]
            final_positions[0][1] = centroid[1]
            final_positions[0][2] = centroid[2]
        elif mesh == "m":
            centroid = np.round(centroid / 0.75,2)
            final_positions[1][0] = centroid[0]
            final_positions[1][1] = centroid[1]
            final_positions[1][2] = centroid[2] 
        elif mesh == "l":
            final_positions[2][0] = centroid[0]
            final_positions[2][1] = centroid[1]
            final_positions[2][2] = centroid[2]

    # print("Initial Positions:") 
    # print(np.round(initial_positions,2))    
    # print("Final Positions:")
    # print(np.round(final_positions,2))  



    # Calculate the rotation matrix using orthogonal Procrustes problem
    U, _, Vt = np.linalg.svd(np.dot(initial_positions.T, final_positions))
    R_matrix = np.dot(U, Vt)

    # Check if the determinant is negative (improper rotation)
    if np.linalg.det(R_matrix) < 0:
        U[:, -1] = -U[:, -1]
        R_matrix = np.dot(U, Vt)

    # Print the rotation matrix
    # print("Rotation Matrix:")
    # print(R_matrix)      
    # Convert the rotation matrix to quaternion
    rotation = R.from_matrix(R_matrix)
    quat = rotation.as_quat()
    quat = quaternion.quaternion(quat[3], quat[0], quat[1], quat[2]) 
    return quat




def get_case( object_name , id,swap=False): 
    
    mesh = get_mesh(object_name,id)
    meshes = mesh.split()
    start_ori_meshes = {}  
    goal_ori_meshes = {}
    for mesh in meshes:
        centroid = mesh.centroid    
        area = mesh.area  
        if area == 122:
            finger_mesh = mesh
            translation_vector = centroid*-1      
            finger_mesh.apply_translation(translation_vector)
        if round(centroid[1]) == 0 and area > 4:
            if centroid[0]<0:
                start_object = mesh   
                start_left_f = round(centroid[0])+18
                start_right_f = round(centroid[0])+18
                start_dist_z = round(centroid[2])  
                translation_vector = centroid*-1      
                start_object.apply_translation(translation_vector)

            else:
                goal_object = mesh  
                no_of_faces = len(goal_object.faces)
                no_of_vertices = len(goal_object.vertices)
                bounding_box = goal_object.bounding_box_oriented
                dimensions = bounding_box.extents  
                dimensions = np.round(dimensions,2)
                dimensions = list(dimensions)   

                goal_left_f = round(centroid[0])-8
                goal_right_f = round(centroid[0])-8
                goal_dist_z = round(centroid[2]) 
                translation_vector = centroid*-1     
                goal_object.apply_translation(translation_vector) 
        if area <4:
            if centroid[0]<0:
                if round(area,2) == 0.69:
                    start_ori_meshes["s"]=mesh.centroid    
                elif round(area,2) == 1.00:
                    start_ori_meshes["m"]=mesh.centroid  
                elif round(area,2) == 1.32:
                    start_ori_meshes["l"]=mesh.centroid  
                elif round(area,2) == 3.12:
                    start_ori_meshes["origin"]=mesh.centroid 
            else:
                if round(area,2) == 0.69:
                    goal_ori_meshes["s"]=mesh.centroid 
                elif round(area,2) == 1.00:
                    goal_ori_meshes["m"]=mesh.centroid 
                elif round(area,2) == 1.32:
                    goal_ori_meshes["l"]=mesh.centroid 
                elif round(area,2) == 3.12:
                    goal_ori_meshes["origin"]=mesh.centroid 
    
    start_ori = quaternion.quaternion(1,0,0,0) 
    goal_ori = find_orientation(goal_ori_meshes, start_ori_meshes)  

    start_palm_width = 0
    goal_palm_width = 0

    start_state = {"left_f":start_left_f, "right_f":start_right_f, "dist_z":start_dist_z, "obj_ori":start_ori, "obj_width":start_palm_width, "rot_angle":0}       
    print("Start State:", start_state)
    goal_state = {"left_f":goal_left_f, "right_f":goal_right_f, "dist_z":goal_dist_z, "obj_ori":goal_ori, "obj_width":goal_palm_width , "rot_angle":0}
    print("Goal State:", goal_state)
    object_data = {"name": "hexagonal_prism", "faces":no_of_faces, "vertices":no_of_vertices, "bounding_box":bounding_box, "dimensions":str(dimensions), "case_no":id}  
    if swap:
        return goal_object, finger_mesh , goal_state, start_state, object_data
    else:
        return start_object , finger_mesh, start_state, goal_state  , object_data      

