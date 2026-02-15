from operator import ne
import trimesh
import numpy as np
import fcl
import os
import pyvista as pv
from scripts.Planner_common_files.class_definitions import VF
import ipdb
from scipy.spatial import ConvexHull
from scripts.Planner_common_files.rotation_data import compute_rotation_angles



class CollsionCheck():
    def __init__(self, finger_mesh):
        self.left_finger_mesh = finger_mesh
        self.right_finger_mesh = finger_mesh
        self.pv_left_finger = pv.wrap(finger_mesh)
        self.pv_right_finger = pv.wrap(finger_mesh)
        self.finger_length = finger_mesh.extents[0]
        self.finger_thickness = finger_mesh.extents[1] +0.2
        self.finger_height = finger_mesh.extents[2]

    def create_collision_object(self, mesh):
        # Extract vertices and faces
        verts = np.array(mesh.vertices, dtype=np.float32)
        tris = np.array(mesh.faces, dtype=np.int32)

        # Create FCL BVHModel
        m = fcl.BVHModel()
        m.beginModel(len(verts), len(tris))
        m.addSubModel(verts, tris)
        m.endModel()

        # Wrap in a collision object
        obj = fcl.CollisionObject(m)
        return obj
    
    def visualize_scene(self, current_state, palm_width, object_mesh):
        
        left_f = current_state.left_f
        right_f = current_state.right_f
        dist_z = current_state.dist_z
        obj_ori = current_state.obj_ori

        # Make a copy of the finger meshes for visualization
        left_finger_mesh = self.pv_left_finger.copy()
        right_finger_mesh = self.pv_right_finger.copy()

        # Compute translations
        left_finger_transform = np.array([-left_f + self.finger_length / 2, palm_width/2 + self.finger_thickness/2, -dist_z])
        right_finger_transform = np.array([-right_f + self.finger_length / 2, -palm_width/2 - self.finger_thickness/2, -dist_z])

        # Apply translation
        left_finger_mesh.translate(left_finger_transform, inplace=True)
        right_finger_mesh.translate(right_finger_transform, inplace=True)

        # translate object mesh to centroid at origin
        object_mesh_centroid = object_mesh.centroid
        # print("Object Centroid:", object_mesh_centroid)
        translation_vector = -object_mesh_centroid
        object_mesh.apply_translation(translation_vector)
        # Apply rotation and translation to object mesh
        # Apply rotation using quaternion
        w, x, y, z = obj_ori.w, obj_ori.x, obj_ori.y, obj_ori.z
        quat = [w, x, y, z]
        matrix = trimesh.scene.transforms.kwargs_to_matrix(quaternion=quat)
        object_mesh.apply_transform(matrix)


        # For object, use quaternion + translation helper
        object_mesh = pv.wrap(object_mesh.copy())

        # Visualization
        plotter = pv.Plotter()
        # Add axes at origin for reference
        axes_actor = pv.AxesActor()
        plotter.add_actor(axes_actor)
        plotter.add_mesh(left_finger_mesh, color='blue', opacity=0.5, label='Left Finger')
        plotter.add_mesh(right_finger_mesh, color='red', opacity=0.5, label='Right Finger')
        plotter.add_mesh(object_mesh, color='green', opacity=0.5, label='Object')
        plotter.add_legend()
        # Set camera position for better view
        plotter.camera_position = 'xy'
        plotter.show()

    def check_collision_fcl(self, current_state, palm_width, object_mesh):
        # Create collision object for the object mesh
        
        obj = self.create_collision_object(object_mesh)
        left_f = current_state.left_f
        right_f = current_state.right_f

        if abs(left_f - right_f) >5 or left_f <=0 or right_f <=0:
            return True

        left_finger_obj = self.create_collision_object(self.left_finger_mesh)
        right_finger_obj = self.create_collision_object(self.right_finger_mesh)
        left_finger_obj.setTranslation(
                np.array([-left_f + self.finger_length / 2,
                        -palm_width/2 - self.finger_thickness/2,
                        0.0], dtype=np.float32)
            )
        right_finger_obj.setTranslation(
                np.array([-right_f + self.finger_length / 2,
                        palm_width/2 + self.finger_thickness/2,
                        0.0], dtype=np.float32)
            )

        # Check collision with left finger
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        # return True if collision detected
        ret1 = fcl.collide(obj, left_finger_obj, request, result)
        if ret1 > 0:
            # print("Collision with left finger")
            return True
        # Check collision with right finger
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        ret2 = fcl.collide(obj, right_finger_obj, request, result)
        if ret2 > 0:
            # print("Collision with right finger")
            return True
        return False
    

    def is_valid_state(self, state, left_most_pt_x, plan_config=None):
        left_most_pt_x +=1
        if state.left_f + left_most_pt_x < VF.x_bounds[0] -3 or state.right_f + left_most_pt_x < VF.x_bounds[0] -3 or state.left_f + left_most_pt_x >= VF.x_bounds[1] or state.right_f + left_most_pt_x >= VF.x_bounds[1]:
            return False
        if abs(state.left_f - state.right_f) > plan_config["max_finger_spread"]:
            return False
        return True
    
    def is_valid_bop_state(self, state, left_most_pt_x, plan_config=None):
        left_most_pt_x +=1
        if state.left_f + left_most_pt_x < VF.x_bounds[0] or state.right_f + left_most_pt_x < VF.x_bounds[0] -3 or state.left_f + left_most_pt_x >= VF.x_bounds[1] or state.right_f + left_most_pt_x >= VF.x_bounds[1]:
            return False
        return True
    

    def visualize_grasp_volume(self, vertices):
        if len(vertices) == 0:
            return
        point_cloud = pv.PolyData(vertices)
        plotter = pv.Plotter()
        axes_actor = pv.AxesActor()
        plotter.add_actor(axes_actor)
        plotter.add_mesh(point_cloud, color='orange', point_size=5, render_points_as_spheres=True)
        # add convex hull around the point cloud
        plotter.camera_position = 'xy'
        plotter.show()


    def calculate_palm_width(self, total_vertices, smallest, current_state):
        # Palm width calculation
        dist_z = current_state.dist_z
        vertices_in_grasp_volume = total_vertices[np.logical_and(total_vertices[:, 2] > VF.z_bounds[0] - dist_z, total_vertices[:, 2] < VF.z_bounds[1] - dist_z)]
        if len(vertices_in_grasp_volume) != 0:
            vertices_in_grasp_volume = vertices_in_grasp_volume[vertices_in_grasp_volume[:, 0] <= (VF.x_bounds[1]-smallest)] 

            if len(vertices_in_grasp_volume) >= 2:
                sorted_vertices = vertices_in_grasp_volume[vertices_in_grasp_volume[:, 1].argsort()]
                palm_width = sorted_vertices[-1][1] - sorted_vertices[0][1]  # grasp volume gives us the palm width
                # print("Palm width: ", palm_width)
                return palm_width
        return None
    
    def check_move_contact_up(self, total_vertices, palm_width, smallest, curr_state):
        # Move contact up/down feasibility
        dist_z = curr_state.dist_z  
        move_grasp_up = total_vertices[np.logical_and(total_vertices[:, 2] > VF.z_bounds[0] - dist_z,  total_vertices[:, 2] < VF.z_bounds[1] -dist_z)]
        bottom_point = total_vertices[np.argmin(total_vertices[:, 2])][2] 
        if bottom_point > VF.z_bounds[0] - dist_z:
            
            return False
        if len(move_grasp_up) > 0:
            move_grasp_up = move_grasp_up[move_grasp_up[:, 0] <= (VF.x_bounds[1]-smallest)] 
            if len(move_grasp_up) == 0:
                return False
            up_sorted_vertices = move_grasp_up[move_grasp_up[:, 1].argsort()]
            palm_width_up = up_sorted_vertices[-1][1] - up_sorted_vertices[0][1]  
            # print("Palm width up: ", palm_width_up)
            if abs(palm_width_up - palm_width) < 0.03:   # 0.03 is the threshold value
                return True
        return False
    
    def check_move_contact_dn(self, total_vertices, palm_width, smallest, curr_state):
        smallest = curr_state.left_f if curr_state.left_f < curr_state.right_f else curr_state.right_f
        dist_z = curr_state.dist_z
        move_grasp_down = total_vertices[np.logical_and(total_vertices[:, 2] > VF.z_bounds[0] - dist_z, total_vertices[:, 2] < VF.z_bounds[1] - dist_z)]
        if len(move_grasp_down) > 0:
            move_grasp_down = move_grasp_down[move_grasp_down[:, 0] <= (VF.x_bounds[1]-smallest)]
            if len(move_grasp_down) == 0:
                return False
            dn_sorted_vertices = move_grasp_down[move_grasp_down[:, 1].argsort()]   
            palm_width_down = dn_sorted_vertices[-1][1] - dn_sorted_vertices[0][1]
            # print("Palm width down: ", palm_width_down)
            if abs(palm_width_down - palm_width) < 0.03:
                move_dn = True
                return True
        return False
    
    def check_slide_left_up(self, total_vertices, palm_width, curr_state):
        smallest = curr_state.left_f if curr_state.left_f < curr_state.right_f else curr_state.right_f
        dist_z = curr_state.dist_z  
        slide_grasp_left_up = total_vertices[np.logical_and(total_vertices[:, 2] > VF.z_bounds[0] - dist_z, total_vertices[:, 2] < VF.z_bounds[1] - dist_z)]
        if len(slide_grasp_left_up) > 0:
            slide_grasp_left_up = slide_grasp_left_up[slide_grasp_left_up[:, 0] <= (VF.x_bounds[1]-smallest)] 
            if len(slide_grasp_left_up) == 0:
                return False, None
            left_sorted_vertices = slide_grasp_left_up[slide_grasp_left_up[:, 1].argsort()]
            palm_width_left = left_sorted_vertices[-1][1] - left_sorted_vertices[0][1]
            # print("Palm width left up: ", palm_width_left)
            if abs(palm_width_left - palm_width) < 0.03:
                leftmost_pt_x = np.min(slide_grasp_left_up[:, 0])
                return True, leftmost_pt_x
        return False, None

    def check_slide_left_down(self, total_vertices, palm_width, curr_state):
        dist_z = curr_state.dist_z  
        smallest = curr_state.left_f if curr_state.left_f < curr_state.right_f else curr_state.right_f
        slide_grasp_left_dn = total_vertices[np.logical_and(total_vertices[:, 2] > VF.z_bounds[0] - dist_z, total_vertices[:, 2] < VF.z_bounds[1] - dist_z)]
        if len(slide_grasp_left_dn) > 0:
            slide_grasp_left_dn = slide_grasp_left_dn[slide_grasp_left_dn[:, 0] <= (VF.x_bounds[1]-smallest)] 
            if len(slide_grasp_left_dn) == 0:
                return False, None
            left_sorted_vertices = slide_grasp_left_dn[slide_grasp_left_dn[:, 1].argsort()]
            palm_width_left = left_sorted_vertices[-1][1] - left_sorted_vertices[0][1]
            # print("Palm width left down: ", palm_width_left)
            if abs(palm_width_left - palm_width) < 0.03:
                leftmost_pt_x = np.min(slide_grasp_left_dn[:, 0])

                return True, leftmost_pt_x
        return False, None
    

    def check_convey_up(self, total_vertices, palm_width, curr_state):
        dist_z = curr_state.dist_z  
        smallest = curr_state.left_f 
        convey_up = total_vertices[np.logical_and(total_vertices[:, 2] > VF.z_bounds[0] - dist_z, total_vertices[:, 2] < VF.z_bounds[1] - dist_z)]
        if len(convey_up) > 0:
            convey_up = convey_up[convey_up[:, 0] <= (VF.x_bounds[1]+smallest)] 
            if len(convey_up) == 0:
                return False
            left_sorted_vertices = convey_up[convey_up[:, 1].argsort()]
            palm_width_after = left_sorted_vertices[-1][1] - left_sorted_vertices[0][1]
            # print("Palm width after convey: ", palm_width_after)
            if abs(palm_width_after - palm_width) < 0.03:
                leftmost_pt_x = np.min(convey_up[:, 0])

                return True, leftmost_pt_x
        return False, None
    

    def check_convey_down(self, total_vertices, palm_width, curr_state):
        dist_z = curr_state.dist_z  
        smallest = curr_state.left_f 
        convey_down = total_vertices[np.logical_and(total_vertices[:, 2] > VF.z_bounds[0] - dist_z, total_vertices[:, 2] < VF.z_bounds[1] - dist_z)]
        if len(convey_down) > 0:
            convey_down = convey_down[convey_down[:, 0] <= (VF.x_bounds[1]-smallest)] 
            if len(convey_down) == 0:
                return False
            left_sorted_vertices = convey_down[convey_down[:, 1].argsort()]
            palm_width_after = left_sorted_vertices[-1][1] - left_sorted_vertices[0][1]
            # print("Palm width after convey: ", palm_width_after)
            if abs(palm_width_after - palm_width) < 0.03:
                leftmost_pt_x = np.min(convey_down[:, 0])

                return True, leftmost_pt_x
        return False, None  


    def check_slide_right_up(self, total_vertices, palm_width, curr_state):
        dist_z = curr_state.dist_z  
        smallest = curr_state.right_f if curr_state.right_f < curr_state.left_f else curr_state.left_f
        slide_grasp_right_up = total_vertices[np.logical_and(total_vertices[:, 2] > VF.z_bounds[0] - dist_z, total_vertices[:, 2] < VF.z_bounds[1] - dist_z)]
        if len(slide_grasp_right_up) > 0:
            slide_grasp_right_up = slide_grasp_right_up[slide_grasp_right_up[:, 0] <= (VF.x_bounds[1]-smallest)] 
            if len(slide_grasp_right_up) == 0:
                return False, None
            right_sorted_vertices = slide_grasp_right_up[slide_grasp_right_up[:, 1].argsort()]
            palm_width_right = right_sorted_vertices[-1][1] - right_sorted_vertices[0][1]
            # print("Palm width right up: ", palm_width_right)
            if abs(palm_width_right - palm_width) < 0.03:
                leftmost_pt_x = np.min(slide_grasp_right_up[:, 0])

                return True, leftmost_pt_x
        return False, None
    
    def check_slide_right_down(self, total_vertices, palm_width, curr_state):
        dist_z = curr_state.dist_z  
        smallest = curr_state.right_f if curr_state.right_f < curr_state.left_f else curr_state.left_f
        slide_grasp_right_dn = total_vertices[np.logical_and(total_vertices[:, 2] > VF.z_bounds[0] - dist_z, total_vertices[:, 2] < VF.z_bounds[1] - dist_z)]
        # print("length of slide_grasp_right_dn: ", len(slide_grasp_right_dn))
        if len(slide_grasp_right_dn) > 0:
            slide_grasp_right_dn = slide_grasp_right_dn[slide_grasp_right_dn[:, 0] <= (VF.x_bounds[1]-smallest)] 
            if len(slide_grasp_right_dn) == 0:
                return False, None
            right_sorted_vertices = slide_grasp_right_dn[slide_grasp_right_dn[:, 1].argsort()]
            palm_width_right = right_sorted_vertices[-1][1] - right_sorted_vertices[0][1]
            # print("Palm width right down: ", palm_width_right)  
            if abs(palm_width_right - palm_width) < 0.03:
                leftmost_pt_x = np.min(slide_grasp_right_dn[:, 0])

                return True, leftmost_pt_x
        return False, None
    
    def find_data_from_mesh(self, total_vertices, current_state, palm_width):
        dist_z = current_state.dist_z  
        smallest = current_state.left_f if current_state.left_f < current_state.right_f else current_state.right_f
        grasp_volume = total_vertices[np.logical_and(total_vertices[:, 2] > VF.z_bounds[0] - dist_z, total_vertices[:, 2] < VF.z_bounds[1] - dist_z)]

        # max x bound for grasp volume
        max_x = grasp_volume[:, 0].max()
   
        if max_x > (VF.x_bounds[1]-smallest):   # if rotation pivot point is out of finger reach
            return  0,0,0,0
        
        grasp_volume = grasp_volume[grasp_volume[:, 0] <= (VF.x_bounds[1]-smallest)] 
        if len(grasp_volume) < 3:
            return  0,0,0,0
        

        # CollsionCheck.visualize_grasp_volume(self, grasp_volume)
        return compute_rotation_angles(grasp_volume, smallest=smallest, slide_resolution=1, visualize=False, palm_width=palm_width)
    
    def check_pivot_horizontal(self, total_vertices, curr_state, palm_width, plan_config):
        # rotate vertices around y axis by 90 degrees without cpnsidering current orientation
        bottom_pt_z = np.min(total_vertices[:,2])
        right_most_pt_x = np.max(total_vertices[:,0])
        dist_z = curr_state.dist_z  
        # print("Bottom pt z:", bottom_pt_z, "dist_Z:", dist_z)
        if bottom_pt_z + dist_z < VF.z_bounds[0] - plan_config["pivot_horizontal_offset"]:  # if the bottom point is below the finger surface we can perform pivot
            pivot_action = "Pivot_Horizontal"
        elif right_most_pt_x + curr_state.left_f > VF.x_bounds[1] + plan_config["pivot_vertical_offset"]:  # if the right most point is out of finger  we can perform pivot
            pivot_action = "Pivot_Vertical"
        else:
            return False, None, None,  None
        rotated_vertices = total_vertices.copy()
        theta = np.radians(90)
        c, s = np.cos(theta), np.sin(theta)
        R_y = np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
        rotated_vertices = rotated_vertices.dot(R_y.T)
        bottom_pt_z = np.min(rotated_vertices[:,2])
        leftmost_pt_x = np.min(rotated_vertices[:,0])
        new_dist_z = round(bottom_pt_z - VF.z_bounds[0])
        smallest = curr_state.left_f if curr_state.left_f < curr_state.right_f else curr_state.right_f
        pivot_horizontal = rotated_vertices[np.logical_and(rotated_vertices[:, 2] > VF.z_bounds[0] - dist_z, rotated_vertices[:, 2] < VF.z_bounds[1] - dist_z)]
        if len(pivot_horizontal) > 0:
            pivot_horizontal = pivot_horizontal[pivot_horizontal[:, 0] <= (VF.x_bounds[1]-smallest)] 
            if len(pivot_horizontal) ==0:
                return False, None, None,   None
            pivot_horizontal = pivot_horizontal[pivot_horizontal[:, 1].argsort()]
            palm_width_left = pivot_horizontal[-1][1] - pivot_horizontal[0][1]
            # print("Palm width left up: ", palm_width_left)
            if abs(palm_width_left - palm_width) < 0.03:
                pivot_horizontal = True

                return True, pivot_action, new_dist_z,leftmost_pt_x
        return False, None, None,   None
    


    def check_rot(self, total_vertices, curr_state, palm_width):
        # rotate vertices around y axis by 90 degrees without cpnsidering current orientation
        rotated_vertices = total_vertices.copy()
        curr_ori = curr_state.obj_ori
        RR = trimesh.transformations.quaternion_matrix([curr_ori.w, curr_ori.x, curr_ori.y, curr_ori.z])[:3, :3]
        rotated_vertices = rotated_vertices.dot(RR.T)
        leftmost_pt_x = np.min(rotated_vertices[:,0])
        # CollsionCheck.visualize_grasp_volume(self, rotated_vertices)
        return True, leftmost_pt_x
        

def subdivide_hull_edges(hull_vertices, step=1.0):
    """
    Subdivide the edges of a convex hull to create intermediate points.
    
    Parameters:
        hull_vertices : np.array, shape (N, 3)  - the vertices of convex hull
        step          : float                  - approximate distance between points along edge

    Returns:
        subdivided_points : np.array of shape (M, 3)
    """
    # make all z values zero
    hull = ConvexHull(hull_vertices)
    points = []

    # Loop through hull edges (simplices)
    for simplex in hull.simplices:
        p1 = hull_vertices[simplex[0]]
        p2 = hull_vertices[simplex[1]]

        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        n_div = max(int(np.ceil(edge_len / step)), 1)  # at least 1 division
        for i in range(n_div + 1):
            point = p1 + (edge_vec * i / n_div)
            points.append(point)

    # Remove duplicates
    subdivided_points = np.unique(np.array(points), axis=0)
    return subdivided_points
 
def normalize_quaternion(q):
    return q / q.norm() 
