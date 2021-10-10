# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:41:56 2020

@author: Yimin Zhang
"""
from scipy.spatial import cKDTree 
import numpy as np
import math
import random
import time
import open3d as o3d
import cv2
from numba import njit
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Downsample point cloud using voxel filter
def downsample(points, voxel_size, mode='Mean'):
    """
    #param mode#
    
    C: set voxel center as the new point
    
    Random: set a random point inside the voxel as the new point
    
    NN: set nearest neighbour of vocel center as the new point 
    
    Mean: set mean value of all the pionts inside the voxel as the new point
    
    """
    if mode != 'C' and  mode != 'Random' and  mode != 'NN' and  mode != 'Mean':
        print('Failure by downsampling')
        return
    
    # Shift point clouds to align with xyz axis
    points_shifted = np.copy(points[:,0:3])
    point_boundary = np.min(points[:,0:3],axis=0)
    points_shifted -= point_boundary

    
    # Give each point a voxel id
    # Voxel_id indicates which voxel each point belongs to
    voxel_coordinate = points_shifted // voxel_size
    [vx_length, vy_length, vz_length] = voxel_coordinate.max(axis=0) + 1
    voxel_id = voxel_coordinate[:, 0] + voxel_coordinate[:, 1]*vx_length + voxel_coordinate[:, 2]*vx_length*vy_length 
    
    # Sort voxel id from small to large
    # Voxel_idx_list includes the index of voxel_id in order from small to large
    voxel_idx_list = np.argsort(voxel_id)
    
    # Order the voxel ids for counting how many points does each voxel have
    voxel_idx_sorted = voxel_id[voxel_idx_list]
    
    # Loop over all voxels
    # Compute new point
    new_points = []
    count1 = 0
    for ii in range(len(voxel_idx_sorted)-1): # Here the last voxel indix is ignored
                                              # How can we include it?
        # continue if they belonge to the same voxel
        if voxel_idx_sorted[ii] == voxel_idx_sorted[ii+1]:
            continue
        
        else:
            # array of point indices including all points belonge to one specific voxel
            # Voxel_idx_idx 
            point_idx = voxel_idx_list[count1:ii+1]
            
            
            # append new points
            if mode == 'C':
                voxel_center = voxel_coordinate[point_idx[0]]*voxel_size + voxel_size/2 + point_boundary
                new_points.append(voxel_center)
            
            elif mode == 'Random':
                new_points.append(points[random.choice(point_idx)])
                
            elif mode == 'NN':
                # search the nearest neighbour of voxel center
                # reture distance and point index of nn
                voxel_center = voxel_coordinate[point_idx[0]]*voxel_size + voxel_size/2
                NN_idx = np.argmin(np.linalg.norm(points_shifted[point_idx] - voxel_center,axis=1))
                new_points.append(points[point_idx][NN_idx])
        
            elif mode == 'Mean':
                new_points.append(np.mean(points[point_idx], axis=0))
        
            # set count1 equle to first point indice of next voxel
            count1 = ii + 1            

    # return new points as array   
    new_points = np.array(new_points)
    return new_points

# Downsampling with farthest point principle
def downsample_FPS(points, n):
    points_new = []
    points_new.append(points[0])
    points_new = np.array(points_new)
    points = np.delete(points,0,axis=0)
    #points_new = points_new[np.newaxis,:]

    for i in range(n-1):
        print(i)
        idx = 0
        d = 0
        for idx_ in range(len(points)):
            d_ = np.min(np.linalg.norm(points[idx_,0:3]-points_new[:,0:3],axis=1))
             
            #print(d_)
            if d_ > d:
                # print(d)
                idx = np.copy(idx_)
                d = np.copy(d_)
        
        points_new = np.concatenate((points_new,points[idx][np.newaxis,:]),axis=0)
        points = np.delete(points,idx,axis=0)
    return points_new


def get_max_cons_matrix(nv_prod, indices, matrix_size, heading_search_size, search_model, use_normal_vectors):
    
    if search_model == 'xy':
        # The axis of matrix is (x, y)
        max_cons_matrix = np.zeros((matrix_size, matrix_size))
        if use_normal_vectors:
            # Use nv product as weight.            
            np.add.at(max_cons_matrix, (indices[:,0], indices[:,1]), nv_prod)
        else:
            # Do not use scalar product as weight, instead fix at 1.0.
            np.add.at(max_cons_matrix, (indices[:,0], indices[:,1]), 1.0)
            
    elif search_model == 'xyz':
        # The axis of matrix is (x, y, z)
        max_cons_matrix = np.zeros((matrix_size, matrix_size, matrix_size))
        if use_normal_vectors:           
            np.add.at(max_cons_matrix, (indices[:,0], indices[:,1], indices[:,2]), nv_prod)
        else:
            np.add.at(max_cons_matrix, (indices[:,0], indices[:,1], indices[:,2]), 1.0)
        
    elif search_model == 'xyheading':
        # The axis of matrix is (x, y, heading)
        max_cons_matrix = np.zeros((matrix_size, matrix_size, heading_search_size))
        if use_normal_vectors:
            # Use appended rotation indices in points_1 as indices for adding up max_cons_matrix            
            np.add.at(max_cons_matrix, (indices[:,0], indices[:,1], indices[:,2]), nv_prod)
        else:
            np.add.at(max_cons_matrix, (indices[:,0], indices[:,1], indices[:,2]), 1.0)
            
    elif search_model == 'xyzheading':
        # The axis of matrix is (x, y, z, heading)
        max_cons_matrix = np.zeros((matrix_size, matrix_size, matrix_size, heading_search_size))
        if use_normal_vectors:
            # Use appended rotation indices in points_1 as indices for adding up max_cons_matrix
            np.add.at(max_cons_matrix, (indices[:,0], indices[:,1], indices[:,2],indices[:,3]), nv_prod)
        else:
            np.add.at(max_cons_matrix, (indices[:,0], indices[:,1], indices[:,2],indices[:,3]), 1.0)

    return max_cons_matrix




# Find indices of outliers out of k sigma
# 数学基础参考https://www.jianshu.com/p/7ea61c9c2135
def find_outlier(points, k=1, covarince_matrix=np.array([[1,0],[0,1]]), user_define = False):
    """
    find indices of outliers that lay beyond the k-sigma Confidence Ellipse of input data
    return index of outliers
    input data can be multi-dimensional
    """
    
    # Check if the input data has the correct format
    if points.shape[1] > points.shape[0]:
        points = points.T
    
    # calculate mean value and subtract it from points
    if not user_define:
        points = points - np.mean(points, axis = 0)
    
    # Calculate the covarince matrix and the eigenvalue & eigenvector of it
    # Eigenvalue is the square of length of semi-major axis of error ellipse
    # Eigenvector is the transformation matrix from initial coordinate to coordinate based on error ellipse 
    if user_define:
        cov_matrix = covarince_matrix
        eig_value, eig_vector = np.linalg.eig(cov_matrix)
        points = points[:,0:cov_matrix.ndim]
    else:
        cov_matrix = np.cov(points.T)
        eig_value, eig_vector = np.linalg.eig(cov_matrix)
    
    # Since eigenvalue is square of semi-major axis, multiple k square
    eig_value *= k**2
    
    # Align points with semi-major axis of error ellipse
    points_ellipse = np.dot(points, eig_vector)
    
    # Build error ellipse
    # The multi-dimentional error ellipse has form x^2/a^2 + y^2/b^2 + ... = 1
    confidence_ellipse = np.diag(np.dot(np.dot(points_ellipse,np.linalg.inv(np.diag(eig_value))),points_ellipse.T))
    
    # Compare points with the error ellipse and return indices of outliers
    # If one point lie outside of error ellipse, its value will be greater than 1
    idx = np.array(range(points_ellipse.shape[0]))
    idx = idx[confidence_ellipse > 1]
    
    return idx


def check_intergrity(cov, shift, highest_point_coordinate, sigma = 1, alert_limit = 0.29):
    cov_matrix = 2*np.linalg.inv(cov[highest_point_coordinate[0],
                                 highest_point_coordinate[1]])
    
    eig_value, eig_vector = np.linalg.eig(cov_matrix)
    idx = np.argsort(eig_value)
    eig_value_sorted = eig_value[idx]   #ein_value from small to large
    eig_vector_sorted = eig_vector[:,idx]
    
    # Check the eccentricity of the error ellipse,
    # which is given by sqrt(1-b^2/a^2)
    eccentricity = np.sqrt(1-eig_value_sorted[0]/eig_value_sorted[1])
    
    
    # Check intergrity
    # Reutrn True if the true error is within the error ellipse
    eig_value_sorted *= sigma**2
    points_ellipse = np.dot(shift[0:2], eig_vector_sorted)
    # print(np.sqrt(eig_value_sorted))
    # print(points_ellipse)
    confidence_ellipse = np.dot(np.dot(points_ellipse,np.linalg.inv(np.diag(eig_value_sorted))),points_ellipse.T)
    intergrity = (confidence_ellipse < 1)
    
    # Check avalability
    # Return True if the error ellipse doesn't exceed the alert_limit
    avalability = (np.max(np.sqrt(eig_value_sorted)) <alert_limit)
    
    
    # Check if the orientation of the error ellipse is consistent with the error
    # Return True if the orientation is NOT consistent with the error
    heading = False
    if eccentricity > 0.5:
        if abs(points_ellipse[1]) < abs(points_ellipse[0]):
            heading = True
    
    
    return eccentricity, intergrity, np.max(np.sqrt(eig_value_sorted)), heading




# mean_shift algorithmus

def Mean_Shift(points, band_width, weight=1, max_iteration = 50, use_weight=True):
    """
    # Use Mean-Shift algorithmus to find all the peaks in max cons matrix
    # Returns coordinate of peaks and index of peaks which each points belongs to 
    # Index doesn't always start from zero
    """
    # Find dimention of input data 
    dim = points.shape[1]
    
    # Initiate all points as one cluster 
    # The axis of 'clusters' are (x, y, cluster_index, state, point_id)
    # State == 0:  Not Converged
    # State == 1: Converged
    # State == 2: Not converged, but has same path with other clusters and don't need to be concerned
    
    clusters = points
    clusters = np.concatenate((clusters, np.arange(len(points))[:,np.newaxis]),axis=1)
    clusters = np.concatenate((clusters, np.zeros(len(points))[:,np.newaxis]),axis=1)
    clusters = np.concatenate((clusters, np.arange(len(points))[:,np.newaxis]),axis=1)
    
    # Initiate archiv to store all visited coordinates
    clusters_archiv = np.empty([0,dim+3])
    
    # Initiate array to store final results
    # The axis of clusters are (x, y, cluster_index)
    cluster_centers = np.empty([0,dim+1])
    
    converge = False
    iteration = 0
    
    # Loop until all clusters converge
    while not converge:
        
        # Loop over all clusters
        for i in range(len(clusters)):
            
            # Pass the converged clusters
            if clusters[i, dim+1] != 0:
                continue

            # Already visited coordinates don't need to be visited again
            # If coordinate already appears in archiv, 
            # conbine two clusters and set state to 2
            bol = np.logical_and.reduce(clusters[i,0:dim] == clusters_archiv[:,0:dim], axis=1)
            if bol.any():
                clusters[i] = clusters_archiv[bol][0]
                clusters[i,dim+1] = 2
                continue
            
            gaussian_kernal = np.exp(-np.power(np.linalg.norm(points-clusters[i,0:dim],axis=1),1)/band_width**2)
            Weight = gaussian_kernal * weight
            #print(Weight)
            # Append old center to archiv so that we can skip over this coordinate in the following loops
            old_center = clusters[i].copy()
            clusters_archiv = np.concatenate((clusters_archiv, old_center[np.newaxis,:]),axis=0)
            
            # Update clusters by calculating weighted mean
            clusters[i,0:dim] = np.around(np.average(points, weights=Weight, axis=0),1)


            # If center doesn't move, append to cluster centers and set state to 1
            if np.array_equal(old_center[0:dim], clusters[i,0:dim]):

                # Set to converged
                clusters[i,dim+1] = 1
                
                # Combine 'same' clusters
                # Calculate distance to all existing converged cluster centers and compare with band width
                distance = np.linalg.norm(cluster_centers[:,0:dim] - clusters[i,0:dim],axis=1)
                
                # Append to cluster centers only wenn no other centers in range of 'band_width'
                if not (distance < band_width).any():
                    cluster_centers = np.concatenate((cluster_centers, clusters[i,0:dim+1][np.newaxis,:]),axis=0)                
                # Otherwise, conbine it to the closest cluster by setting the cluster index 
                else:
                    clusters[i,0:dim+1] = cluster_centers[np.argmin(distance), 0:dim+1]
        
        
        # Break if cluters is empty or max iteration is reached
        iteration += 1
        
        if  np.logical_and.reduce(clusters[:,dim+1]!=0, axis=0) or iteration == max_iteration:
            converge = True
            if iteration == max_iteration:
                print("Shit")
                clusters[clusters[i, dim+1]==0] = np.append(cluster_centers[0],(1,0))

    # Set state 2 points to its corresponding clusters
    bol = clusters[:,dim+1]==2
    update_cluster_list = np.argwhere(bol)
    
    # clusters_tmp = clusters[bol]
    for idx in update_cluster_list:
        clusters = update_cluster(idx, clusters, dim, update_cluster_list)
        
    
    # if np.logical_or.reduce(bol):
    #     clusters_tmp = clusters[clusters[bol][:,dim+2].astype(int)]
    #     clusters[bol] = clusters_tmp
    
    # Return cluster centers and cluster index        
    cluster_idx = clusters[:,dim]
    
    return cluster_centers, cluster_idx


# Sub-Function of func Mean_Shift()
def update_cluster(cluster_idx, clusters, dim, pose):
    
    # Pass updated clusters
    # State == 1 indicates cluster alrealy updated
    if clusters[cluster_idx, dim+1] == 1:
        return clusters
    
    # Check if the goal layer also need to be updated
    # Update clusters at lower layer firstly 
    elif np.logical_or.reduce(clusters[cluster_idx, dim+2] == pose):
        update_cluster(clusters[cluster_idx, dim+2].astype(int), clusters, dim, pose)
        
    clusters[cluster_idx] = clusters[clusters[cluster_idx, dim+2].astype(int)]
    
    # Return updated clusters
    return clusters


# 
def get_matrix_score(peak_number, peak_value_difference, percentage, 
                     peak_score=1, diff_score=1, prec_score=1):
    
    score = 0
    
    if peak_number == 1:
        return score
    
    else:
        score += peak_score
        if percentage < 1.5/peak_number:
            score += prec_score

        if peak_value_difference >= 0.95:
            score += diff_score*(peak_value_difference-0.95)*100
        
    return score




# Find the peak of max_cons_matrix and calculate the corresponding shift

def calc_shift(max_cons_matrix, max_cons_radius, max_cons_grid_edge, 
               max_cons_heading_range, max_cos_grid_angle, heading_search_radius, search_model):
     
    max_idx = np.argmax(max_cons_matrix)
    if search_model == 'xy':
        # Flip the matrix so that "x" is to the right and "y" is up.
        max_cons_matrix = np.flipud(max_cons_matrix.T)
        # np.savetxt('fname' + str(count) + '_nv.txt',max_cons_matrix)
        shift = np.array(divmod(max_idx, max_cons_matrix.shape[1])) * max_cons_grid_edge - max_cons_radius
    else:
        # For 3 dimentional max cons matrix, firstly find z index
        z_idx, residual = divmod(max_idx, max_cons_matrix[0].size)
        if search_model == 'xyz':
            z_shift = z_idx * max_cons_grid_edge - max_cons_radius
        else: 
            z_shift = (z_idx - heading_search_radius) * max_cos_grid_angle
            # ensure z_shift is inside max_cons_heading_range
            z_shift = max(min(z_shift, max_cons_heading_range), -max_cons_heading_range)
            if search_model == 'xyzheading':
                z_idx, residual = np.array(divmod(residual, max_cons_matrix[0,0].size))
                z_shift1 = z_idx * max_cons_grid_edge - max_cons_radius
        # Then find xy index same as by 2 dimentional matrix case and compute shift
        shift = np.array(divmod(residual, max_cons_matrix.shape[max_cons_matrix.ndim-1])) * max_cons_grid_edge - max_cons_radius
        shift = np.append(shift, [z_shift])
        if search_model == 'xyzheading':
            shift = np.insert(shift, 2, [z_shift1])
            
    # The columns of shift is (x, y, z, heading, peak value)
    shift = np.append(shift, np.max(max_cons_matrix))
    
    
    return max_cons_matrix, shift


# Find matched points on map 
def find_matched_points(map_points, scan_points, idx_map, idx_scan, indice, shift, max_cons_radius, max_cons_grid_edge, z_scale):
    
    # Since z axis is scaled up duo to the search process,
    # Make a copy of map points to scale down z to its orginal coordinate 
    map_temp = map_points.copy()
    map_temp[:,2] /= z_scale
    scan_points[:,2] /= z_scale
    
    # Indice contains all shift between matched pairs
    # The index where shift equal to max cons shift are the index of points we are looking for 
    # Convert indice to real world coordinate
    # Compare it with 'shift'
    indice = indice * max_cons_grid_edge - max_cons_radius
    idx = np.logical_and.reduce(indice-shift[0:2] ==0, axis=1)
    # Also compare indice with (0,0) to get correctly matched points
    idx2 = np.logical_and.reduce(indice==0, axis=1)
    
    # Return founded points
    matched_map_points = map_temp[idx_map][idx]
    matched_scan_points = scan_points[idx_scan][idx]
    correct_map_points = map_temp[idx_map][idx2]
    return matched_map_points, matched_scan_points, correct_map_points


# Generate an array including all rotated points
#@njit
def calc_rotated_points(points, max_cons_heading_range, max_cos_grid_angle, heading_search_size, r_center):
    
    # Generate an array including all rotation angles
    # Firstly rotate point cloud to the very left, then to the right 
    # If there is an unexact division, append the remainder to both side of the array
    if max_cons_heading_range % max_cos_grid_angle == 0:
        angle_set = np.append([-max_cons_heading_range], [max_cos_grid_angle for i in range(heading_search_size-1)])
    else:
        angle_set = np.append([-max_cons_heading_range, max_cons_heading_range%max_cos_grid_angle], 
                              [max_cos_grid_angle for i in range(heading_search_size-3)])
        angle_set = np.append(angle_set,[max_cons_heading_range%max_cos_grid_angle])

    # Generat an empty array to store rotated points
    # The columns of array are ("world_x", "world_y", "world_z", "nx", "ny", "nz", "rotation_idx")
    points_extended = np.empty((0,7))              
    
    # Create point cloud with open3d
    points_rot = o3d.geometry.PointCloud()
    points_rot.points = o3d.utility.Vector3dVector(points[:,0:3])
    
    
    for rotation_idx in range(heading_search_size):   
               
        # Build rotation matrix and rotate the point could alone z axis
        # Note the rotation angle is given in degree and needs to be convered to radian
        rotation_angle = angle_set[rotation_idx]
        
        R = points_rot.get_rotation_matrix_from_xyz((0,0,rotation_angle*math.pi/180))
        points_rot.rotate(R, center=r_center)  #### Here we need to define rotation center!
        
        # Also rotate the normal vectors
        points[:,3:6] = np.dot(points[:,3:6],R.T)
        
        # Convert Open3D.o3d.geometry.PointCloud to numpy array
        points_numpy1 = np.asarray(points_rot.points)
        
        # Extend columns 
        # The columns of array are ("world_x", "world_y", "world_z", "nx", "ny", "nz", "rotation_idx")
        rotation_idx_array = np.ones((points_numpy1.shape[0],1)) * rotation_idx
        points_with_rotation_idx = np.concatenate((points_numpy1, points[:,3:6]),axis=1)
        points_with_rotation_idx = np.concatenate((points_with_rotation_idx,rotation_idx_array),axis=1)

        # Extend rows
        # Put all rotated points in one array
        points_extended = np.concatenate((points_extended,points_with_rotation_idx),axis=0)
           
    return points_extended

# Rotate points alone z axis
def rotate_points(points, angle, rotation_center):
    points_ = points[:,0:3] - rotation_center
    angle_ = angle*np.pi/180
    R = np.array([[np.cos(angle_), np.sin(angle_), 0],
                  [-np.sin(angle_), np.cos(angle_), 0],
                  [0             , 0             , 1]])
    
    points_ = np.dot(points_, R)
    points_ += rotation_center
    points_ = np.concatenate((points_, np.dot(points[:,3:6],R)),axis=1)
    return points_
    
    
    
# Find the main line in the array using Hough Transformation
# The coordinates in the Hough Transformation satisfy
# rho = x*cos(theta) + y*sin(theta)
def Hough_Transformation(points, weights, angle_grid_size=1, use_weight = True):
    """
    Find the main line in the array using hough transformation
    Return parameters of the maximum consensus line
    """
    # Ensure points only have axis (x,y)
    points = points[:,0:2]
    
    # Make an array including theta-coordinate in hough space
    thetas = np.arange(0, 180, angle_grid_size)/180*np.pi
    
    # Duplicate weights with length of thetas
    # Ensure weights are positive
    weights[weights<0] = 0

    weights = np.repeat(weights, len(thetas))
    
    # Make an array to store transformed points in hough space
    hough_space_coordinates = np.empty((0,2))

    # Iterate all points
    for point in points:
        
        # Calculate coorespongding rho-coordinate in hough space
        # Note the coordinates in the Hough Transformation satisfy
        # rho = x*cos(theta) + y*sin(theta)
        x, y = point
        rhos = x*np.cos(thetas) + y*np.sin(thetas)
        coordinates = np.concatenate((thetas[:,np.newaxis],rhos[:,np.newaxis]), axis=1)
      
        # Add to array
        hough_space_coordinates = np.concatenate((hough_space_coordinates,coordinates),axis=0)
    
    # Shift rho by its minimum and shift back after search process
    shift = np.min(hough_space_coordinates[:,1])
    hough_space_coordinates[:,1] -= shift
    
    # Scale thetas down to angle
    hough_space_coordinates[:,0] *= 180/np.pi

    # Create matrix in hough space and add points to it
    hough_space_matrix = np.zeros((len(thetas), np.max(hough_space_coordinates[:,1]).astype(int)+1))  
    if use_weight:
        np.add.at(hough_space_matrix, (hough_space_coordinates[:,0].astype(int), hough_space_coordinates[:,1].astype(int)), weights)
    else:
        np.add.at(hough_space_matrix, (hough_space_coordinates[:,0].astype(int), hough_space_coordinates[:,1].astype(int)), 1)
    
    # Find the maximum consensus line 
    line_idx = np.argmax(hough_space_matrix)
    theta, rho = np.array(divmod(line_idx, hough_space_matrix.shape[1]))
    
    # Shift rho back to its original value
    # Find cooresponding theta and convert theta to radian
    # Note the theta in Hough space is orthogonal to the line
    # Here we subtract pi/2 from it
    rho += shift
    theta *= angle_grid_size/180*math.pi
    theta -= math.pi/2
    
    # Also find other potential lines by using mean shift in theta-axis
    peak_value = np.max(hough_space_matrix)
    pos = hough_space_matrix > 0.8*peak_value
    line_parameters = np.argwhere(pos)[:,0]
    # center, idx = Mean_Shift(line_parameters[:,np.newaxis], 45, use_weight=False)
    # line_number = len(np.unique(idx))
    # if line_number>1:
    #     center_list = np.unique(center[:,0])
    #     if (center_list[0]-center_list[1]+180)<45:
    #         line_number -= 1
    
    # Return parameters of the maximum consensus line
    return theta, rho#, line_number
    

def calc_covarince_score_matrix(points_0, points_1, matched_points_idx, matrix_size, heading_search_size, search_model):
    """
    Calculate the size of error ellipse for each cell in maximun consensus matrix
    by building covarince matrix of the cell with normal vectors
    """
    
    # Make a list of all matched points
    # The colume of matched_points_idx contains
    # the indices of each matched pair and there correspounding position in the accumulator.
    # Which is
    # ("map_point_index", "scan_point_index", "matrix_x", "matrix_y", "matrix_z", "matrix_heading")
    matched_map_points = points_0[matched_points_idx[:,0].astype(int)]
    matched_scan_points = points_1[matched_points_idx[:,1].astype(int)]
    
    # Compute cosin of normal vectors and use as weights
    # Note cos(theta) = n1*n2/norm(n1)/norm(n2)
    nv_prod = np.sum(matched_map_points[:,3:6] * matched_scan_points[:,3:6], axis=1)
    nv_prod[nv_prod < 0.0] = 0.0
    costheta = nv_prod
    
    # Compute each component of covarince matrix
    # Take only normel vector in x-achse and y-achse
    # Note the columns of matched_map_points contains global coordinates and normal vectors.
    # which is ("world_x", "world_y", "world_z", "nx", "ny", "nz")
    cov_xx = matched_map_points[:,3]*matched_map_points[:,3]*costheta**2
    cov_yy = matched_map_points[:,4]*matched_map_points[:,4]*costheta**2
    cov_xy = matched_map_points[:,3]*matched_map_points[:,4]*costheta**2
    
    
    # Make covarince matrix and accumulate all components.
    # Note covarince matrix is symmetry 
    if search_model == 'xy':
        # The axis of matrix is (x, y, covarince matrix)
        cov_matrix = np.zeros((matrix_size,matrix_size,2,2))
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],0,0), cov_xx)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],0,1), cov_xy)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],1,1), cov_yy)
        cov_matrix[:,:,1,0] = cov_matrix[:,:,0,1]
        cov_matrix[:,:,0,0] += 1e-10
        cov_matrix[:,:,1,1] += 1e-10
        
        
    elif search_model == 'xyz' or search_model == 'xyheading':
        if search_model == 'xyz':
            # The axis of matrix is (x, y, z, covarince matrix)
            cov_matrix = np.zeros((matrix_size,matrix_size,matrix_size,2,2))
        else:
            # The axis of matrix is (x, y, heading, covarince matrix)
            cov_matrix = np.zeros((matrix_size,matrix_size,heading_search_size,2,2))    
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],0,0), cov_xx)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],0,1), cov_xy)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],1,1), cov_yy)
        cov_matrix[:,:,:,1,0] = cov_matrix[:,:,:,0,1]
        cov_matrix[:,:,:,0,0] += 1e-10
        cov_matrix[:,:,:,1,1] += 1e-10
        
        
    elif search_model == 'xyzheading':
        # The axis of matrix is (x, y, z, heading, covarince matrix)
        cov_matrix = np.zeros((matrix_size,matrix_size,matrix_size,heading_search_size,2,2))
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],matched_points_idx[:,5],0,0), cov_xx)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],matched_points_idx[:,5],0,1), cov_xy)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],matched_points_idx[:,5],1,1), cov_yy)
        cov_matrix[:,:,:,:,1,0] = cov_matrix[:,:,:,:,0,1]
        cov_matrix[:,:,:,:,0,0] += 1e-10
        cov_matrix[:,:,:,:,1,1] += 1e-10
    
    
    # Estimate size of error ellipse by computing a^2 + b^2,
    # where a and b are the semi-major axis of error ellipse, 
    # which is then the eigenvalue of inv(cov).
    # Note a^2 + b^2 = trace(inv(cov)) = trace(cov)/det(cov).
    cov_score_matrix = np.trace(cov_matrix,axis1=cov_matrix.ndim-2,axis2=cov_matrix.ndim-1) / np.linalg.det(cov_matrix)
    
    # Fit the results by elementweise inverse so that large value corresponds to low uncertenty
    cov_score_matrix = 1./cov_score_matrix
        
    return cov_score_matrix



def Hough_space_accumulator(points_0, points_1, matched_points_idx, matrix_size, heading_search_size, search_model, angle_grid_size=1, Normalization=False):
    """Build Hough space accumulator
       for points belong to every cell in maximum consensus matrix"""
    
    # Make a list of all matched points
    # The colume of matched_points_idx is 
    # ("map_point_index", "scan_point_index", "matrix_x", "matrix_y", "matrix_z", "matrix_heading")
    matched_points = points_0[matched_points_idx[:,0].astype(int)]
    matched_points[:,0:3] -= np.min(matched_points[:,0:3],axis=0)
    
    orientation_size = 180//angle_grid_size 
    
    # Calculate orientation of normal vectors and convert into degree
    # Take only normel vector in x-achse and y-achse
    # Note the columns of points are ("world_x", "world_y", "world_z", "nx", "ny", "nz")
    # Ensure orientation lay between 0 and pi
    orientation = np.arctan2(matched_points[:,4], matched_points[:,3])
    orientation[orientation<0] += np.pi
    
    # Calculate rho and theta in Hough space
    # Note in Hough space, rho = x*cos(theta) + y*sin(theta)
    rho = (matched_points[:,0]*np.cos(orientation) + matched_points[:,1]*np.sin(orientation)).astype(int)
    orientation = np.floor(orientation*orientation_size/np.pi).astype(int)
    

    # Normalize rho and orientation with their standard deviation
    # Set to true for scaling 
    if Normalization: 
        var_rho = np.sum((rho-np.mean(rho))**2)/len(rho)
        var_orientation = np.sum((orientation-np.mean(orientation))**2)/len(orientation)
        rho = (10*(rho-np.mean(rho)) / var_rho**0.5).astype(int)
        orientation = (10*(orientation-np.mean(orientation)) / var_orientation**0.5).astype(int)
    
    # Shift rho and orientation by its minimum
    rho -= np.min(rho)
    orientation -= np.min(orientation)
    
    # Make covarince matrix and accumulate all components.
    # Note covarince matrix is symmetry 
    if search_model == 'xy':
        # The axis of matrix is (x, y, distance, orientation)
        histogram_matrix = np.zeros((matrix_size,matrix_size,np.max(rho)+1,np.max(orientation)+1))
        np.add.at(histogram_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],rho,orientation), 1)

    elif search_model == 'xyz' or search_model == 'xyheading':
        if search_model == 'xyz':
            # The axis of matrix is (x, y, z, distance, orientation)
            histogram_matrix = np.zeros((matrix_size,matrix_size,matrix_size,np.max(rho)+1,orientation_size))
        else:
            # The axis of matrix is (x, y, heading, distance, orientation)
            histogram_matrix = np.zeros((matrix_size,matrix_size,heading_search_size,np.max(rho)+1,np.max(orientation)+1))  
        np.add.at(histogram_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],rho,orientation), 1)    
        
    elif search_model == 'xyzheading':
        # The axis of matrix is (x, y, z, heading, distance, orientation)
        histogram_matrix = np.zeros((matrix_size,matrix_size,matrix_size,heading_search_size,np.max(rho)+1,orientation_size))
        np.add.at(histogram_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],matched_points_idx[:,5],rho,orientation),1) 
        
    # Note here we compute the histogram of normal vector,
    # which is orthogonal of the line
    return histogram_matrix



def Entropy(data, axis=-1):
    
    # Reshape the input data to be one dimensional
    data = data.reshape(-1)
    
    # Normalize data by deviding sum of it
    # Ensure the normalized input data not equal to 0
    data_normalized = data/np.sum(data)
    data_normalized[data_normalized==0] += 1e-10
    
    entropy = -np.sum(data_normalized*np.log2(data_normalized))
    
    return entropy
    

def calc_cross_covarince_score_matrix(points_0, points_1, matched_points_idx, matrix_size, heading_search_size, search_model):
    """
    Calculate the size of error ellipse for each cell in maximun consensus matrix
    by building covarince matrix of the cell with normal vectors
    """
    points_1[:,0:2] -= np.mean(points_1[:,0:2],axis=0)
    #points_1[:,0:2] /= np.max(np.abs(points_1[:,0:2]),axis=0)
    # Make a list of all matched points
    # The colume of matched_points_idx is 
    # ("map_point_index", "scan_point_index", "matrix_x", "matrix_y", "matrix_z", "matrix_heading")
    matched_map_points = points_0[matched_points_idx[:,0].astype(int)]
    matched_scan_points = points_1[matched_points_idx[:,1].astype(int)]
    
    # Compute cosin of normal vectors and use as weights
    # Note cos(theta) = n1*n2/norm(n1)/norm(n2)
    nv_prod = np.sum(matched_map_points[:,3:5] * matched_scan_points[:,3:5], axis=1)
    nv_prod[nv_prod < 0.0] = 0.0
    nv_len = np.linalg.norm(matched_map_points[:,3:5],axis=1)*np.linalg.norm(matched_scan_points[:,3:5],axis=1)
    costheta = nv_prod/nv_len
    
    # Compute each component of covarince matrix
    # Take only normel vector in x-achse and y-achse
    # Note the columns of matched_map_points are ("world_x", "world_y", "world_z", "nx", "ny", "nz")
    cross_cov_Xnx = np.abs(matched_scan_points[:,0]*matched_map_points[:,3]*costheta**2)
    cross_cov_Yny = np.abs(matched_scan_points[:,1]*matched_map_points[:,4]*costheta**2)
    cross_cov_Xny = np.abs(matched_scan_points[:,0]*matched_map_points[:,4]*costheta**2)
    cross_cov_Ynx = np.abs(matched_scan_points[:,1]*matched_map_points[:,3]*costheta**2)
    
    # Make covarince matrix and accumulate all components.
    # Note covarince matrix is symmetry 
    if search_model == 'xy':
        # The axis of matrix is (x, y, covarince matrix)
        cov_matrix = np.zeros((matrix_size,matrix_size,2,2))
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],0,0), cross_cov_Xnx)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],0,1), cross_cov_Xny)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],1,1), cross_cov_Yny)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],1,0), cross_cov_Ynx)
        cov_matrix[:,:,0,0] += 1e-10
        cov_matrix[:,:,1,1] += 1e-10
        
        
    elif search_model == 'xyz' or search_model == 'xyheading':
        if search_model == 'xyz':
            # The axis of matrix is (x, y, z, covarince matrix)
            cov_matrix = np.zeros((matrix_size,matrix_size,matrix_size,2,2))
        else:
            # The axis of matrix is (x, y, heading, covarince matrix)
            cov_matrix = np.zeros((matrix_size,matrix_size,heading_search_size,2,2))    
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],0,0), cross_cov_Xnx)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],0,1), cross_cov_Xny)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],1,1), cross_cov_Yny)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],1,1), cross_cov_Ynx)
        cov_matrix[:,:,:,0,0] += 1e-10
        cov_matrix[:,:,:,1,1] += 1e-10
        
        
    elif search_model == 'xyzheading':
        # The axis of matrix is (x, y, z, heading, covarince matrix)
        cov_matrix = np.zeros((matrix_size,matrix_size,matrix_size,heading_search_size,2,2))
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],matched_points_idx[:,5],0,0), cross_cov_Xnx)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],matched_points_idx[:,5],0,1), cross_cov_Xny)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],matched_points_idx[:,5],1,1), cross_cov_Yny)
        np.add.at(cov_matrix, (matched_points_idx[:,2],matched_points_idx[:,3],matched_points_idx[:,4],matched_points_idx[:,5],1,1), cross_cov_Ynx)
        cov_matrix[:,:,:,:,0,0] += 1e-10
        cov_matrix[:,:,:,:,1,1] += 1e-10
    
    
    #cov_matrix = np.linalg.pinv(cov_matrix)
    # Estimate size of error ellipse by computing a^2 + b^2,
    # where a and b are the semi-major axis of error ellipse, 
    # which is then the eigenvalue of inv(cov).
    # Note a^2 + b^2 = trace(inv(cov)) = trace(cov)/det(cov).
    cov_score_matrix = np.trace(cov_matrix,axis1=cov_matrix.ndim-2,axis2=cov_matrix.ndim-1) / np.linalg.det(cov_matrix)
    
    # Fit the results by elementweise inverse so that large value corresponds to low uncertenty
    cov_score_matrix = 1./cov_score_matrix
        
    return cov_score_matrix


# Open 3d implementation of ICP
def ICP_o3d(source, target, rotation_center, shift, z_scale, search_model, max_disp=0.5, max_step=40):

    # Initialize scan and map point cloud by applying correction 
    # based on maximum consensus approach
    source[:,0] += shift[0]
    source[:,1] += shift[1]
    target[:,2] /= z_scale
    scan_pc = o3d.geometry.PointCloud()
    map_pc = o3d.geometry.PointCloud()
    scan_pc.points = o3d.utility.Vector3dVector(source[:,0:3])
    map_pc.points = o3d.utility.Vector3dVector(target[:,0:3])
    
    # Inteplate results from max_cons_approach
    init_disp = np.identity(4)
    # init_disp[0,3] = shift[0]
    # init_disp[1,3] = shift[1]
    # if search_model == 'xyz':
    #     init_disp[2,3] = shift[2]
    # if search_model == 'xyheading' or search_model == 'xyzheading':
    #     init_disp[2,3] = shift[2]
    #     init_disp[2,2] = shift[3]
    
    
    # Do ICP 
    icp_p2p = o3d.pipelines.registration.registration_icp(scan_pc, map_pc, max_disp, init_disp, 
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(), 
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_step))#设置最大迭代次数

    # Transform results 
    # Use rotation_center as control point
    rotation_center[0] += shift[0]
    rotation_center[1] += shift[1]
    rotation_center = np.append(rotation_center, 1)
    results = np.dot(icp_p2p.transformation, rotation_center[:,np.newaxis]) - rotation_center[:,np.newaxis]
        
    return results[0:3]

# Use matched map points only
def ICP_normal_vectors(scan_, map_, shift, z_scale, search_model, rotation_center, threshold=0.0001, max_step=40, use_weight=True):
    
    # Initialize scan and map point cloud by applying correction 
    # based on maximum consensus approach
    if search_model == 'xyheading':
        scan_ = rotate_points(scan_, shift[2], rotation_center)
    elif search_model == 'xyzheading':
        scan_ = rotate_points(scan_, shift[3], rotation_center)
    scan_[:,0] += shift[0]
    scan_[:,1] += shift[1]
    
    
    t=np.zeros((2,1))
    ans = t
    step = 0
    Iteration = True
    while Iteration:
        
        # Update scan point cloud
        scan_[:,0] += t[0]
        scan_[:,1] += t[1]
        
        # Find unique index of matched_scan_points
        # Note if one point occurs multipel time in scan points, 
        # np.unique will return only the index of its first occurrence
        # Sort the lndex from small to large
        _, unique_indices = np.unique(scan_, return_index=True, axis=0)
        unique_indices = unique_indices[np.argsort(unique_indices)]
        
        unique_idx_list =[]
        ini = 0
        for i in unique_indices:
            
            # First element in unique_indices is always 0, should be ignored.
            if i == ini:
                continue
            
            # Neighboured index indicate unique points in original matched scan point cloud, 
            # no further prosess is needed, its index will be kept.
            elif i - ini ==1:
                unique_idx_list.append(ini)
                ini = i
            
            # Unneighboured index indicate multipel map points correspond to same scan point. 
            # In this case, only the closest point will be kept.
            else:
                m_points_tmp = map_[range(ini,i)][:,0:3]
                s_points_tmp = scan_[range(ini,i)][:,0:3]

                closest_points_idx = np.argmin(np.linalg.norm(m_points_tmp-s_points_tmp,axis=1))
                unique_idx_list.append(ini+closest_points_idx)
                ini = i
 
        # Reutrn unique point set
        scan_1 = scan_[unique_idx_list]
        map_1 = map_[unique_idx_list]
        

        
        # Compute cosine of normal vectors and use as weights
        # Note cos(theta) = n1*n2/norm(n1)/norm(n2)
        nv_prod = np.sum(scan_1[:,3:6] * map_1[:,3:6], axis=1)
        nv_prod[nv_prod < 0.0] = 0.0
        W = np.diag(nv_prod)
    
        # A Matrix contains all normal vectors of map points on xy-plane
        A = map_1[:,3:5]
        
        # L Matrix contains all observations 
        L = np.sum((map_1[:,0:2] - scan_1[:,0:2]) * A, axis=1)[:,np.newaxis]
    
     
        if use_weight:
            # If use weigt applies, compute weighted Least Square
            t = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.T, W),A)), A.T),W), L)

        else:
            # Otherweise, compute Conventional Least Square
            t = np.dot(np.dot(np.linalg.inv(np.dot(A.T ,A)), A.T), L)
            
            
        ans += t
        step += 1
        
        if np.linalg.norm(t) < threshold or step > max_step:
            Iteration = False
            
    return ans



# Use points of whole map point cloud
def ICP_normal_vectors2(tree_0, scan_, map_, shift, z_scale, search_model, rotation_center,threshold=0.0001, max_step=40, use_weight=True):
    
    # Initialize scan and map point cloud by applying correction 
    # based on maximum consensus approach
    if search_model == 'xyheading':
        scan_ = rotate_points(scan_, shift[2], rotation_center)
    elif search_model == 'xyzheading':
        scan_ = rotate_points(scan_, shift[3], rotation_center)
    scan_[:,0] += shift[0]
    scan_[:,1] += shift[1]
    
    # Scale up z of scan points for the tree search
    scan_[:,2] *= z_scale
    # Scale down z back to its original value 
    map_[:,2] /= z_scale
    
    
    
    
    
    
    t=np.zeros((2,1))
    ans = t
    step = 0
    Iteration = True
    while Iteration:
        
        # Update scan point cloud
        scan_[:,0] += t[0]
        scan_[:,1] += t[1]
        
        # Carry out NN tree search,
        # which will return the index of nearest neighbour for each scan point
        _, b = tree_0.query(scan_[:,0:3])

        # Nearest map points for each scan points
        map_1 = map_[b]

        
        # Compute cosine of normal vectors and use as weights
        # Note cos(theta) = n1*n2/norm(n1)/norm(n2)
        nv_prod = np.sum(scan_[:,3:6] * map_1[:,3:6], axis=1)
        nv_prod[nv_prod < 0.0] = 0.0
        W = np.diag(nv_prod)
    
        # A Matrix contains all normal vectors of map points on xy-plane
        A = map_1[:,3:5]
        
        # L Matrix contains all observations 
        L = np.sum((map_1[:,0:2] - scan_[:,0:2]) * A, axis=1)[:,np.newaxis]
    
     
        if use_weight:
            # If use weigt applies, compute weighted Least Square
            t = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.T, W),A)), A.T),W), L)

        else:
            # Otherweise, compute Conventional Least Square
            t = np.dot(np.dot(np.linalg.inv(np.dot(A.T ,A)), A.T), L)
            
            
        ans += t
        step += 1
        
        if np.linalg.norm(t) < threshold or step > max_step:
            Iteration = False
            
    return ans

# Create LoG Kernel for the Blob detecter
def createLoGKernel(sigma, size):
    # Unpack higth and width of kernal
    H, W = size
    
    # Create grid on x and y direction
    r, c = np.mgrid[0:H:1.0, 0:W:1.0]
    r -= (H-1)/2
    c -= (W-1)/2
    
    # Compute σ^2 and σ^4
    sigma2 = np.power(sigma, 2.0)
    sigma4 = np.power(sigma, 4.0)
    
    # Compute x^2 + y^2
    norm2 = np.power(r, 2.0) + np.power(c, 2.0)
    
    # Generate LoG kernal, which is the second derivative of Gaussian Kernal
    # Note we use here the normallized LoG kernal,
    # which has the form (x^2+y^2-2*σ^2)/(2pi*σ^4)*exp(-(x^2+y^2)/2σ^2)
    LoGKernel = ((norm2-2*sigma2)/(2*np.pi*sigma4))*np.exp(-norm2/(2*sigma2))  # 省略掉了常数系数 1\2πσ4
    
    return LoGKernel


def Calc_blob_radius(matrix, step_width = 1, sigma_max = 5, kernal_side_length=31):
    
    kernal_side_length = 6*sigma_max + 1 
    # kernal_side_length = 61
    
    peak = np.argwhere(matrix == np.max(matrix))[0]
    
    sigma = 1
    kernal_size = [kernal_side_length, kernal_side_length]
    r = int((kernal_side_length-1)/2)
    blob_score = np.array([])

    matrix_ = matrix[peak[0]-r:peak[0]+r+1, peak[1]-r:peak[1]+r+1]
    
    while sigma <= sigma_max:
        
        LoGKernel = createLoGKernel(sigma, kernal_size)
        a = matrix_*LoGKernel
        
        score = np.sum(matrix_*LoGKernel)
        blob_score = np.append(blob_score, score)
        
        sigma += step_width
    
    max_sigma = np.argmin(blob_score)*step_width + 1
    # print(blob_score)
    radius = np.sqrt(2)*max_sigma
    
    return max_sigma


def Calc_blob_radius2(matrix, step_width = 1, sigma_max = 10, kernal_side_length_max=31):
    
    kernal_side_length = 6*sigma_max + 1 
    
    # if kernal_side_length > kernal_side_length_max:
    #     kernal_side_length = kernal_side_length_max

    
    peak = np.argwhere(matrix == np.max(matrix))[0]
    
    sigma = 1
    kernal_size = [kernal_side_length, kernal_side_length]
    r = int((kernal_side_length-1)/2)
    
    
    blob_score = np.empty([1,matrix.shape[0],matrix.shape[1]])

    
    while sigma <= sigma_max:
        
        LoGKernel = createLoGKernel(sigma, kernal_size)
        a = cv2.filter2D(matrix, -1, LoGKernel)
        
        blob_score = np.concatenate((blob_score, a[np.newaxis,:]),axis=0)
        
        sigma += step_width 
    
    max_sigma,x,y = np.argwhere(blob_score==np.min(blob_score))[0]
    shift = [x-peak[0],y-peak[1]]
     #print(x,y)
    if np.linalg.norm(shift) > 7:
        max_sigma = Calc_blob_radius(matrix, step_width, sigma_max, kernal_side_length)
        x=peak[0]
        y=peak[1]
        shift = [0,0]
        
    max_sigma = max_sigma*step_width + 1

   
    radius = np.sqrt(2)*max_sigma
    
    
    
    return np.array([max_sigma,x,y]), shift



def draw_histogram(data, begin, end, step_width, decimal=2):
    
    """
        data: 1d array for histogram
        begin: begin of the array
        
    """
    reverse = False
    if begin > end:
        o = begin
        begin = end
        end = o
        reverse = True
    
    data = np.array(data)
    data = np.round(data,decimal)
    length = int(np.ceil(np.abs((end-begin))/step_width))+1

    aaa = []
    for i in range(length):
        c = begin+i*step_width
        
        # Set upper and down boundary
        c_min = c - 0.5*step_width
        c_max = c + 0.5*step_width
        if i == 0:
            c_min = begin 
        elif i == length:
            c_max = end
            
        # c_min <= data < c_max
        c1 = (c_min <= data)[:,np.newaxis]
        c2 = (data < c_max)[:,np.newaxis]
        cc = np.concatenate((c1,c2),axis=1)
        aaa.append(np.sum(np.logical_and.reduce(cc,axis=1)))
        
    if reverse:
        aaa.reverse()
        
    return aaa



def plot_3d_accumulator(shift, search_model="xyheading"):
    """
    Plot 3d ccumulator
    """
    
    shift_uinque = np.unique(shift[:,0:3],axis=0)
    x=shift_uinque[:,0]
    y=shift_uinque[:,1]
    z=shift_uinque[:,2]
    
    
    # Count occurrence for each cell
    occurrecnce = []
    for i in shift_uinque:
        occurrecnce.append(np.sum(np.logical_and.reduce(shift[:,0:3]==i,axis=1)))
    
    
    # Generate figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pnt3d=ax.scatter(x,y,z,c=occurrecnce,s=8)
    cbar=plt.colorbar(pnt3d)
    
    # Set coordinate interval
    if True:
        x_major_locator=plt.MultipleLocator(0.1)
        y_major_locator=plt.MultipleLocator(0.1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
    
    cbar.set_label("Occurrence")
    plt.xlim(-0.1,0.1)
    plt.ylim(-0.1,0.1)
    ax.set_zlim(-1, 1)
    plt.show()
    
    
    
    
    