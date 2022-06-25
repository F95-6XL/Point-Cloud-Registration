# Alignment of scans using a KDT and numpy.
#
# (c) 01 NOV 2020 Claus Brenner
#
from enum import ENUM
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import open3d as o3d
import glob
import natsort
from function import downsample, get_max_cons_matrix, calc_covarince_score_matrix, calc_rotated_points, ICP_normal_vectors2
from save import save_restuls_to_local
from MaxConsMatrix import MaxConsMatrix



# ---
# Note the columns in the measurement files are:
# ("world_x", "world_y", "world_z", "nx", "ny", "nz",
#  "scan_id", "segment_id", "grid_xy", "reflectance")
# ---



# ---
# Supported search dimentions:
# xy: search in xy plane
# xyz: search in xyz space
# xyheading: search in xy plane including rotation alone z axis
# xyzheading: search in xyz space including rotation alone z axis
# ---
class search_mode(ENUM):
    xy_search = 'xy'
    xyz_search = 'xyz'
    xyheading_search = 'xyheading'
    xyzheading_search = 'xyzheading'

# Hack: helper to draw a "crosshair" to a matrix, using NaN's.
def plot_crosshair(m):
    s = m.shape[0]//2
    m[s,:s-20] = np.nan
    m[s,s+20:] = np.nan
    m[:s-20,s] = np.nan
    m[s+20:,s] = np.nan
    return m

# Compute max cons alignment for two scans.
def align_two_scans(tree_0,points_0, points_1,
        max_cons_radius, max_cons_grid_edge, max_cons_z_range, max_cos_grid_angle, 
        max_cons_heading_range, rotation_center, count, use_normal_vectors=True, search_model = 'xy'):
    
    if search_model not in ['xy', 'xyz', 'xyheading', 'xyzheading']:
        raise ValueError("Unsupported search mode")
        
        
    # In order to have the same search radius in all 3 dimenstions, we
    # simply scale up z.
    z_scale = max_cons_radius / max_cons_z_range
    # points_0[:,2] *= z_scale
    points_1[:,2] *= z_scale
    
    # Calculate matrix_radius and heading_search_radius
    # The max cons matrix will be from -matrix_radius to +matrix_radius,
    # Same by heading
    matrix_radius = int(np.ceil(max_cons_radius / max_cons_grid_edge))
    matrix_size = 2*matrix_radius+1
    heading_search_radius = int(np.ceil(max_cons_heading_range / max_cos_grid_angle))
    heading_search_size = 2 * heading_search_radius + 1
    
    
    if search_model == 'xyheading' or search_model == 'xyzheading':
        # If heading search applies, generate an array including all rotated points
        # The columns of array after rotation are 
        # ("world_x", "world_y", "world_z", "nx", "ny", "nz", "rotation_idx")
        points_1 = calc_rotated_points(points_1, max_cons_heading_range, 
                                      max_cos_grid_angle, heading_search_size, rotation_center)

    
    # For each point in points_1, find all the neighbors.
    # Note we use Minkowski inf-norm, using p=inf.
    # Neighbors will be an array of len(points_1) lists, each list containing
    # the indices of all neighbors. Lists may be empty.
    neighbors = tree_0.query_ball_point(points_1[:,0:3], max_cons_radius,
                                        p=np.inf)

    # # Make a flat list of indices into points_1.
    repetitions = [len(e) for e in neighbors]
    idx_1 = np.repeat(np.arange(len(points_1)), repetitions)

    # Also make a flat list of indices into points_0.
    idx_0 = np.concatenate(neighbors).astype(int)
    # print("pairs: %7d" % idx_0.shape[0])
    
    # Compute all xy offsets.
    # Here we compute the shift of scan 1, relative to scan 0.
    if search_model == 'xyz' or search_model == 'xyzheading':
        offsets = points_1[:,0:3][idx_1] - points_0[:,0:3][idx_0]
    else:
        offsets = points_1[:,0:2][idx_1] - points_0[:,0:2][idx_0]      

    # If required, compute scalar products of normal vectors.
    if use_normal_vectors:
        nv_prod = np.sum(points_0[:,3:6][idx_0] * points_1[:,3:6][idx_1],
                          axis=1)
        # Zero where it is negative.
        nv_prod[nv_prod < 0.0] = 0.0
    
    # Convert into bins and matrix indices.
    factor = matrix_radius / max_cons_radius
    indices = np.round(offsets * factor).astype(int) + matrix_radius
    if search_model == 'xyheading' or search_model == 'xyzheading':
        # If heading search applys, also append rotation index to indices list
        indices = np.concatenate((indices, points_1[:,6][idx_1][:,np.newaxis]),axis=1).astype(int)
    
    
    # Find matched points for each cell in the accumulator
    # matched_points_idx includes points index and there corresponding matrix coordinate
    # The colume of matched_points is 
    # ("map_point_index", "scan_point_index", "matrix_x", "matrix_y", "matrix_z", "matrix_heading")
    matched_points_idx = np.concatenate((idx_0[:,np.newaxis], idx_1[:,np.newaxis]), axis=1)
    matched_points_idx = np.concatenate((matched_points_idx, indices), axis=1)
    
    # Make accumulator
    # Here we have two basic approaches:
    # 1. Use count of point matches as score to build up accumulator, 
    #    which is called version 1.0 
    # 2. Use trace of covariance matrix as score to build up accumulator, 
    #    which is called version 2.0 
    if True:
        accumulator = get_max_cons_matrix(nv_prod, indices, matrix_size, heading_search_size, 
                              search_model, use_normal_vectors)
    else:
        accumulator = calc_covarince_score_matrix(points_0, points_1, matched_points_idx, matrix_size, 
                                    heading_search_size, search_model)


    # Save maximum consensus matrix
    if False:
        save_restuls_to_local(accumulator, search_model, count)

   
    Accumulator = MaxConsMatrix(accumulator, max_cons_radius, max_cons_grid_edge, 
                        matrix_size, max_cons_heading_range, max_cos_grid_angle, 
                        heading_search_radius, search_model, z_scale)
    Accumulator.calc_shift()
    
    # Posibility of further optimization
    # Do an extra ICP based on outcomes of maximum consensus approach
    if False:
        matched_map_points_unique, matched_scan_points_unique, _, _ = Accumulator.find_matched_points(points_0, points_1, matched_points_idx[:,0],
                                                                  matched_points_idx[:,1],matched_points_idx[:,2:5], unique=True)
    
        results_icp2 = ICP_normal_vectors2(tree_0, matched_scan_points_unique, points_0.copy(),
                                          -Accumulator.shift, z_scale, search_model, rotation_center, use_weight=True)
        b = Accumulator.shift[0:2] - results_icp2[0:2].flatten()
    
    return accumulator, Accumulator.shift[0:3]
    



    
# ------
# Main
# ------
if __name__ == "__main__":
    


    # ------
    # Parameter settings
    # ------
    
    # Search model, default mode is 'xy'
    search_model = search_mode.xy_search.value
    
    # Angle to select normal vectors. This is used to delete ground points.
    up_vector_angle =10.0
    cos_ub = np.cos(np.deg2rad(up_vector_angle))
    cos_lb = np.cos(np.deg2rad(180.0 - up_vector_angle))

    # Radius of max cons search in meter. Result will be from -radius to +radius.
    max_cons_radius = 1.0

    # The grid size of the max cons grid.
    max_cons_grid_edge = 0.02

    # In this implementation, z is aligned already. When aligning x/y,
    # we will search only for corresponting points in a range +/- zrange.
    max_cons_z_range = 0.05
    if search_model == 'xyz' or search_model == 'xyzheading':
        max_cons_z_range = max_cons_radius
    
    # Angle in degree and grid size of max cons search if search of heading applies
    max_cons_heading_range = 1
    max_cos_grid_angle = 0.25
    
    # Whether to weight the counts using the normal vector scalar product.
    use_normal_vectors = True
    
    # Remove ground
    remove_ground = True
    
    # Downsampling of measurement points in meter, set to 0 to skip downsampling
    voxel_size = 0.2
        
    # if search in heading direction applies, load rotation center
    rotation_center = np.loadtxt('*.txt', delimiter=",")[:,2:5]

    

    

    # ------
    # Read Map
    # ------
    pcl_map_load = o3d.io.read_point_cloud("*.ply")
    pcl_map_load.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=100))
    
    pcl_map_numpy = np.asarray(pcl_map_load.points)
    pcl_map_numpy_normals = np.asarray(pcl_map_load.normals)
    map_pcl=np.concatenate((pcl_map_numpy, pcl_map_numpy_normals),axis=1)
    
    h_map_pcl = map_pcl[np.logical_and(map_pcl[:,5] >= cos_lb, map_pcl[:,5] <= cos_ub)]


    # Trick for kd tree:
    # - In x and y, we want to find all points within distance
    #   +/- max_cons_radius.
    # - However, in z, we want to find points only within max_cons_z_range.
    # In order to have the same search radius in all 3 dimenstions, we
    # simply scale up z.
    z_scale = max_cons_radius / max_cons_z_range
    h_map_pcl[:,2] *= z_scale
    
    tree_ = cKDTree(h_map_pcl[:,0:3])

    # Set to True to visualize the map point could 
    if False:
        o3d.open3d.visualization.draw_geometries([pcl_map_load],point_show_normal=True)
    
    
    
    # ------
    # Read Measurements
    # ------
    path = '*.txt'  
    files = glob.glob(path) 
    files = natsort.natsorted(files)

        
    # Create container to store results over all epoches
    if search_model == 'xy':
        max_cons_shift = np.empty((0, 3))
    elif search_model == 'xyzheading':
        max_cons_shift = np.empty((0, 5))
    else:
        max_cons_shift = np.empty((0, 4))
        
        
        
    # ------
    # Start to loop over all epoches
    # ------
    for count, name in enumerate(files): 
        
        # Read measurements
        points = np.loadtxt(name, delimiter=",")
        
        points_pcd = o3d.geometry.PointCloud()
        points_pcd.points = o3d.utility.Vector3dVector(points)
        points_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.7, max_nn=10000))

    
        # Convert Open3D.o3d.geometry.PointCloud to numpy array
        points_numpy = np.asarray(points_pcd.points)
        normals_numpy = np.asarray(points_pcd.normals)
        points=np.concatenate((points_numpy, normals_numpy),axis=1)
    
        # Vertical surfaces = horizontal normals.
        if remove_ground:
            h_points = points[np.logical_and(
                points[:,5] >= cos_lb, points[:,5] <= cos_ub)]
            # h_points = h_points[h_points[:,2]>98]
            # print("  remaining points on vertical surfaces", len(h_points))
        else:
            h_points=points

        
        # Downsampling of measurement points
        hd_points = downsample(h_points, voxel_size, mode='NN')

            
        print('Epoch:' + str(count))
        
        # Set to True to get a list of all scan id's in this tile.
        if False:
            # scan_id is column 6.
            scan_ids, counts = np.unique(h_points[:,6], return_counts=True)
            print("  sid: count")
            for scan_id, point_count in zip(scan_ids, counts):
                print("  %3d: %5d" % (scan_id, point_count))
    
    
    
        # Call function to align map and measurement using maximum consensus technique
        mc, shift = align_two_scans(tree_,h_map_pcl,hd_points, 
                             max_cons_radius, max_cons_grid_edge, max_cons_z_range,
                             max_cos_grid_angle, max_cons_heading_range, rotation_center[name],
                             count, use_normal_vectors, search_model)
        



        max_cons_shift = np.append(max_cons_shift, [shift], axis=0)
        plt.imshow(plot_crosshair(mc), interpolation="nearest")
        
        # Set to True to save results.
        if False: 
            plt.savefig("./Results/Velodyne/LOD_2/img_"+str(count)+".png")
            


    # Finally, call show.
    plt.show()
