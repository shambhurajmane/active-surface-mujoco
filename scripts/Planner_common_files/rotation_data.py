import numpy as np
import matplotlib.pyplot as plt

# ---------- Simple 3D Point class ----------
class gPoint:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z
    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

def sub(p1, p2):
    return gPoint(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)

def dot(p1, p2):
    return p1.x*p2.x + p1.y*p2.y + p1.z*p2.z

def norm(p):
    return np.linalg.norm([p.x, p.y, p.z])



def find_slide_values(ca,cra,cca,ccra):
    # sliding values are the rotation of point cra about origin for angle ca    
    # sliding valu for clockwise rotation
    # if cra.x < 0:
    #     ca = -ca
    # if ccra.x > 0:
    cca = -cca
    # rotate_pt_z = cra[0] * np.sin(ca) + cra[2] * np.cos(ca)
    # rotate_pt_x = cra[0] * np.cos(ca) - cra[2] * np.sin(ca)
    # rotate about z axis
    rotated_pt_x = cra.x*np.cos(ca) - cra.y*np.sin(ca)
    cw_slide_value = abs(cra.x) + abs(rotated_pt_x)
    # sliding value for counter clockwise rotation

    rotate_pt_x = ccra.x*np.cos(cca) - ccra.y*np.sin(cca)
    ccw_slide_value = abs(ccra.x) + abs(rotate_pt_x)
    # import ipdb; ipdb.set_trace()
    return cw_slide_value, ccw_slide_value


# ---------- Core function ----------
def compute_rotation_angles(vertices_in_grasp_volume, smallest=0.1, slide_resolution=1, visualize=True, palm_width=None):
    sorted_vertices = vertices_in_grasp_volume[vertices_in_grasp_volume[:, 1].argsort()]
    sorted_vertices = np.array(sorted_vertices)

    # make z coordinate zero for all points
    sorted_vertices[:,2] = 0
    
    left_sort = sorted_vertices[sorted_vertices[:, 1] == sorted_vertices[0][1]]  
    max_left_edge_length = max(left_sort[:, 0]) - min(left_sort[:, 0])
    bottom_left_pt = gPoint(min(left_sort[:, 0]), left_sort[0][1], 0)   
    top_left_pt = gPoint(max(left_sort[:, 0]), left_sort[0][1], 0)      

    right_sort = sorted_vertices[sorted_vertices[:, 1] == sorted_vertices[-1][1]]  
    max_right_edge_length = max(right_sort[:, 0]) - min(right_sort[:, 0])
    bottom_right_pt = gPoint(min(right_sort[:, 0]), right_sort[0][1], 0)
    top_right_pt = gPoint(max(right_sort[:, 0]), right_sort[0][1], 0)

    if max(max_left_edge_length, max_right_edge_length) > palm_width +1.5:
        # print("Object cannot be rotated further, returning zeros")
        return 0,0,0,0

    threshold = 0.1
    bottom_pts = np.where(
    np.logical_and.reduce((
        sorted_vertices[:, 0] < 0,
        sorted_vertices[:, 1] > sorted_vertices[0][1]+threshold,    
        sorted_vertices[:, 1] < sorted_vertices[-1][1]-threshold    
        ))
    ) 
    top_pts = np.where(
    np.logical_and.reduce((
        sorted_vertices[:, 0] > 0,
        sorted_vertices[:, 1] > sorted_vertices[0][1]+threshold,    
        sorted_vertices[:, 1] < sorted_vertices[-1][1]-threshold   
        ))
    )    
    if len(bottom_pts[0])==0 or len(top_pts[0])==0:
        return 0,0,0,0
    blnext_pt = gPoint(sorted_vertices[bottom_pts[0][0]][0] , sorted_vertices[bottom_pts[0][0]][1], sorted_vertices[bottom_pts[0][0]][2])
    brnext_pt = gPoint(sorted_vertices[bottom_pts[0][-1]][0] , sorted_vertices[bottom_pts[0][-1]][1], sorted_vertices[bottom_pts[0][-1]][2]) 

    tlnext_pt = gPoint(sorted_vertices[top_pts[0][0]][0], sorted_vertices[top_pts[0][0]][1], sorted_vertices[top_pts[0][0]][2])
    trnext_pt = gPoint(sorted_vertices[top_pts[0][-1]][0], sorted_vertices[top_pts[0][-1]][1], sorted_vertices[top_pts[0][-1]][2])     
    
    # find angle to nest surface
    ###### 4 angles i=need to be calculated viz
    ###### angle till next surface on left finger for rotating clockwise
    ###### angle till next surface on right finger for rotating clockwise
    ###### angle till next surface on left finger for rotating counter clockwise
    ###### angle till next surface on right finger for rotating counter clockwise
    ###### angle for rotating clockwise or counter clockwise will be the minimum of the two angles

    # Clockwise rotation angles for left finger
    v1 = sub(blnext_pt, bottom_left_pt)
    v2 = gPoint(-1, 0,0)
    angle1 = np.arccos(dot(v1, v2)/(np.linalg.norm([v1.x, v1.y, 0])*np.linalg.norm([v2.x, v2.y, 0])))

    # Clockwise rotation angles for right finger
    v1 = sub(trnext_pt, top_right_pt)
    v2 = gPoint(1, 0,0)
    angle2 = np.arccos(dot(v1, v2)/(np.linalg.norm([v1.x, v1.y, 0])*np.linalg.norm([v2.x, v2.y, 0])))

    if angle1 < angle2:
        ca = angle1                                                         # clockwise angle
        cra = bottom_left_pt                                               # clockwise rotation axis
    else:
        ca = angle2
        cra = top_right_pt

    # Counter Clockwise rotation angles for right finger
    # print("brnext_pt", brnext_pt, "bottom_right_pt", bottom_right_pt)       
    v1 = sub(brnext_pt, bottom_right_pt)
    v2 = gPoint(-1, 0,0)
    angle1 = np.arccos(dot(v1, v2)/(np.linalg.norm([v1.x, v1.y, 0])*np.linalg.norm([v2.x, v2.y, 0])))

    # Counter Clockwise rotation angles for left finger
    # print("tlnext_pt", tlnext_pt, "top_left_pt", top_left_pt)   
    v1 = sub(tlnext_pt, top_left_pt)    
    v2 = gPoint(1, 0,0)
    angle1 = np.arccos(dot(v1, v2)/(np.linalg.norm([v1.x, v1.y, 0])*np.linalg.norm([v2.x, v2.y, 0])))

    if angle1 < angle2:
        cca = angle1                                                                        # counter clockwise angle
        ccra = bottom_right_pt              # counter clockwise rotation axis
    else:
        cca = angle2
        ccra = top_left_pt

    # make angle in multiple of 10 degrees for smoother rotation
    ca = np.rad2deg(ca)
    ca= np.round(ca/10)*10    
    ca = np.deg2rad(ca)
    cca = np.rad2deg(cca)
    cca = np.round(cca/10)*10
    cca = np.deg2rad(cca)
    # print("Clockwise angle (deg):", np.rad2deg(ca), "Counter Clockwise angle (deg):", np.rad2deg(cca))
    # if np.rad2deg(ca)<15:
    #     ca=0
    # if np.rad2deg(cca)<15:
    #     cca=0

    cw_slide, ccw_slide = find_slide_values(ca,cra,cca,ccra)

    ca = np.rad2deg(ca)
    ca= np.round(ca)    
    ca = np.deg2rad(ca)
    cca = np.rad2deg(cca)
    cca = np.round(cca) 
    cca = np.deg2rad(cca)


    # slide resolution is 0.5 mm so I want to round the ccw and cw slide values to nearest 0.5 mm
    cw_slide = round(cw_slide*2)/2
    ccw_slide = round(ccw_slide*2)/2
    # print("slide values - cw:", cw_slide, "ccw:", ccw_slide)

    return ca, cca, cw_slide, ccw_slide

# ---------- Example: rectangular prism top view ----------
if __name__ == "__main__":
    rect_vertices = [
        [-1.0, 0.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0]
    ]
    compute_rotation_angles(rect_vertices, visualize=True)
