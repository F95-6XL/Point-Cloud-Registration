# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 00:04:26 2021

@author: Yimin Zhang
"""

import numpy as np
import cv2
import math as math
from scipy import stats
from functions import Mean_Shift, Hough_Transformation


# Postprocessing of the maximum consensus matrix
class MaxConsMatrix:
    
    
    def __init__(self, Matrix, max_cons_radius, max_cons_grid_edge, matrix_size,
                    max_cons_heading_range, max_cos_grid_angle, 
                    heading_search_radius, search_model, z_scale):
        
        # The Maximum Consensus Matrix
        self.Matrix = Matrix
        
        # Constant settings
        self.max_cons_radius = max_cons_radius
        self.max_cons_grid_edge = max_cons_grid_edge
        self.matrix_size = matrix_size
        self.max_cons_heading_range = max_cons_heading_range
        self.max_cos_grid_angle = max_cos_grid_angle
        self.heading_search_radius = heading_search_radius
        self.search_model = search_model
        self.z_scale = z_scale
        
        # Highest Peak value
        self.highest_point_value = np.max(self.Matrix)
        
        
            
    
    
    def calc_shift(self):
        """Find the shift to the highest point and convert to the global coordinate.
           return:
                  self.highest_point_coordinate = [x, y, z, heading]
                       contains parameters with highest consensus set in accumulator coordinate
                       
                  self.shift = [world_x, world_y, world_z, world_heading, peak value]
                       contains parameters with highest consensus set in global coordinate
        """
        
        # Find coordinate of point that have the highest matrix value
        max_points = np.argwhere(self.Matrix == self.highest_point_value)
    
        
        if len(max_points) == 1:
            # If has only one highest matrix value,
            # then it is the point corresponds to shift that we are looking for.
            # Highest_point_coordinate has the axis (x, y, z, heading) in accumulator coordinate
            # Convert its coordinate into global coordinate
            self.highest_point_coordinate = max_points[0]  
            self.shift = self.highest_point_coordinate * self.max_cons_grid_edge - self.max_cons_radius
            
            if self.search_model == 'xyheading' or self.search_model == 'xyzheading':
                # If heading search applies, also find the shift in heading axis
                # Note we need to ensure headding_shift is inside max_cons_heading_range
                # Change the last elemet to heading index
                heading_shift = self.highest_point_coordinate[self.Matrix.ndim-1]*self.max_cos_grid_angle-self.max_cons_heading_range
                heading_shift = max(min(heading_shift, self.max_cons_heading_range), - self.max_cons_heading_range)
                self.shift[self.Matrix.ndim-1] = heading_shift
                
        else:
            # If exists points with same matrix value, 
            # use percentage of peaks to determin main shift
            # Set critical_peak_threshold to 1 to find all peaks with same value
            # The length of critical_peaks is the number of points that have same matrix value
            self.find_critical_points()
            self.find_peaks(critical_peak_threshold=1)
            same_value_peak_number = len(self.critical_peaks)
       
            # If the points that have same matrix value are from the same peak,
            # set this peak to main peak
            if same_value_peak_number == 1:
                main_peak = self.critical_peaks[0,0:2]                
                
            # Else, use percentage of peaks to determin main peak
            # The colume of critical peaks are 
            # (x, y, peak_value, difference to highest peak, percentage in candidate points)
            else:
                main_peaks = self.critical_peaks[0:same_value_peak_number,:]
                main_peak = main_peaks[np.argmax(main_peaks[:,4])]

            # Convert shift into matrix coordinate 
            # Note that only (x,y) coordinate of peaks are stored in critical peaks
            self.highest_point_coordinate = np.rint((main_peak[0:2] + self.max_cons_radius)/self.max_cons_grid_edge).astype(int)
            if self.search_model == 'xy':
                # Main peak is the (x,y) coordinate of shift
                self.shift = main_peak[0:2]
                
            else:
                # If z or heading search is also required,
                # we need to find matrix index in z or heading axis as well,
                # which is not considered in self.critical_peaks.
                # Find the cooresponding points in max cons matrix and read coordinate
                # Note here only the (x,y) coordinates are checked
                self.highest_point_coordinate = max_points[np.logical_and.reduce(max_points[:,self.Matrix.ndim-2:self.Matrix.ndim]
                                                                                 ==self.highest_point_coordinate,axis=1)][0]    
                # Convert matrix coordinate into real world coordinate
                self.shift = self.highest_point_coordinate * self.max_cons_grid_edge - self.max_cons_radius
                
                # If heading search applies, also find the shift in heading axis
                # Note we need to ensure headding_shift is inside max_cons_heading_range
                if self.search_model == 'xyheading' or self.search_model == 'xyzheading':
                    heading_shift = self.highest_point_coordinate[self.Matrix.ndim-1]*self.max_cos_grid_angle-self.max_cons_heading_range
                    heading_shift = max(min(heading_shift, self.max_cons_heading_range), - self.max_cons_heading_range)
                    self.shift[self.Matrix.ndim-1] = heading_shift

        # The columns of shift is (x, y, z, heading, highest point value)
        self.shift = np.append(self.shift, self.highest_point_value)
        
        # Flip the matrix so that "x" is to the right and "y" is up.
        Matrix_flipud = np.flipud(self.Matrix.T)
        return Matrix_flipud
    
   
    
    def find_critical_points(self, p = 0.8, wondow_width = 11, sigma = 3):
        """ Find points in the Matrix that has value larger than p*highest_point_value.
            The search will only be carried out in xy plane.
            Window width and sigma are parameters of smoothing.
            The columes if candidate_points is (x, y, matrix_value, matrix_value after smoothing).
        """
        # Compare matrix value with threshold 
        # Make a boolean array with position of candidate points as True
        pos = self.Matrix > p*self.highest_point_value
        
        # Find candidate points
        self.critical_points = np.argwhere(pos)
        # Take only the x,y coordinate
        # The matrix itself has colume (x, y, z, heading)
        self.critical_points = self.critical_points[:,0:2]  
        
        # Find the corresponding matrix value of candidate points and conbine them together   
        matrix_value = self.Matrix[pos]
        self.critical_points = np.concatenate((self.critical_points,matrix_value[:,np.newaxis]),axis=1)    
    
        # Also append matrix value after smooting to it
        # Here we use Gaussian Filter to smooth the maxtrix value
        # This is the preparation of following procedures 
        # Note we use border type reflection101, so that the border will be treated evenly
        # The columes if candidate_points is (x, y, matrix_value, matrix_value after smoothing)
        self.matrix_filtered = cv2.GaussianBlur(self.Matrix, (wondow_width,wondow_width), sigma, borderType=cv2.BORDER_REFLECT101)
        matrix_value_filtered = self.matrix_filtered[pos]
        self.critical_points = np.concatenate((self.critical_points,matrix_value_filtered[:,np.newaxis]),axis=1)
        
    
    
    def calc_critical_points_center(self, Use_Weight = False):
        """ Calculate the center of candidate points.
            Set use_weight to True to calculate center weighted by matrix value.
        """
        if Use_Weight:
            self.critical_points.center = np.average(self.critical_points[:,0:2], weights=self.critical_points[:,2], axis=0)   
        else:
            self.critical_points.center = np.mean(self.critical_points[:,0:2], axis=0)
    
    
   
    def find_peaks(self, search_wondow = 0.5, critical_peak_threshold = 0.95, Smoothing = True):
        """ Find all geographical peaks in the matrix.
            Set Use Smoothing to True to reduce local extremum.
            Also extract features from peaks.
        """
        # Convert search window to matrix coordinate
        search_wondow = int(np.floor(search_wondow/self.max_cons_grid_edge))
        
        
        # Interplate max cons matrix as a topological surface, use Mean shift algorithmus to find peaks of the surface
        # Reuturn coordinate of peaks and index of peaks to which each candidate points belongs
        # Use matrix value as weights, this helps mean shift algorithmus converge to the geographical peaks 
        # Set Use Smoothing to True to use smoothed matrix value as weights and reduce local extremum
        if  Smoothing:
            peaks, peak_idx = Mean_Shift(self.critical_points[:,0:2],  search_wondow, weight=self.critical_points[:,3])
        else:
            peaks, peak_idx = Mean_Shift(self.critical_points[:,0:2],  search_wondow, weight=self.critical_points[:,2])        

        # Find peak index that the highest point belongs to
        #self.highest_point_idx = peak_idx[self.critical_points[:,2]==self.highest_point_value][0]
        unique_peak_index = np.unique(peak_idx)
        element_number = len(peak_idx)

  
        
        # Local peak will be seen as critical peak if its difference to the highest peak >= critical_peak_threshold
        # Make an array of all critical peaks
        # Note that the critical peaks will include the highest peak itself
        # The colume of critical peaks are 
        # (x, y, peak_value, difference to highest peak, percentage in candidate points)
        self.critical_peaks = np.empty([0, 5])
        
        
        # Loop over all local peaks
        for i in unique_peak_index:
            
            # Store all critical points belong to current peak in 'local_peak_points'
            # Note the critical points have same order as peak index
            local_peak_points = self.critical_points[peak_idx==i]
            
            # Find the local maximum of current peak
            local_peak_value = np.max(local_peak_points[:,2])
            difference = local_peak_value / self.highest_point_value

            # Judgement of critical peak
            if difference >= critical_peak_threshold:
                # Find coordinate of local peak, 
                # here we ignore the case that two points in one peak may have same value.
                # Convert coordinate into global coordinate
                # The colume of local peak are 
                # (x, y, peak_value, difference to highest peak, percentage in candidate points)
                local_peak_coordinate = local_peak_points[local_peak_points[:,2]==local_peak_value][0,0:2]
                local_peak = local_peak_coordinate * self.max_cons_grid_edge - self.max_cons_radius
                local_peak = np.append(local_peak, local_peak_value)
                local_peak = np.append(local_peak, np.array([difference]))
                percantage = np.sum(peak_idx==i) / element_number
                local_peak = np.append(local_peak, percantage)
                
                # Add to critical_peaks
                self.critical_peaks = np.append(self.critical_peaks, [local_peak], axis=0)
                
        
        # Sort the critical peaks from the highest to the lowest
        self.critical_peaks = self.critical_peaks[np.argsort(-self.critical_peaks[:,2])]
        
        # The number of critical peaks is the length of the array - 1
        self.critical_peak_number = self.critical_peaks.shape[0] - 1
        
        
        # Count the peak number in all
        # Here we ignore peak with less than 2 elements
        for i in unique_peak_index:
            if np.sum(peak_idx==i) <= 2 and len(np.unique(peak_idx))!=1:
                peak_idx = np.delete(peak_idx, np.where(peak_idx==i))
        self.peak_number = len(np.unique(peak_idx))
        
        
        
        
    def Watershed(self, sigma=1, step_width=0.025, stop_at='bottom', save_image = False):
        """
        Apply watershed to the accumulator from top to bottom.
        
        Parameters
        ----------
        sigma : float, optional
            sigma of Gaussian filter. The default is 1.
        step_width : float between 0 and 1, optional
            step width that the water level moves. The default is 0.025.
        stop_at : string, optinal
            stop_at = 'bottom': return information of all peaks inclusive their mass and position.
            stop_at = 'second_peak': return only the position of water level of second peak.
            stop_at = float between 0 and 1: stop at the given water level, 
                                             return only the position of water level of second peak.
        """
        
        # Convert Matrix to a list
        # The colum of list is (x, y, matrix value)
        # Convolve matrix with Gaussian filter
        # Note here we use zero padding at border
        self.matrix_filtered = cv2.GaussianBlur(self.Matrix, (11,11), sigma, borderType=cv2.BORDER_ISOLATED)
        matrix_value = self.matrix_filtered.flatten()

        
        # Points has the axis(matrix_value, cluster_index)
        # Cluster_index is initialized to be 0
        # Cluster_index ==  0 : Point not yet assigned
        # Cluster_index  >  0 : Points already assigned, value indicates its cluster
        points = np.concatenate((matrix_value[:,np.newaxis],np.zeros((len(matrix_value),1))),axis=1)


        # Make a list of neighbours
        # Note here we use 8-er neighbours
        neighbours_mask = np.array((-self.matrix_size-1, -self.matrix_size, -self.matrix_size+1,
                                    -1, 1, 
                                    self.matrix_size-1, self.matrix_size, self.matrix_size+1))
        
        # Initialization
        water_level = 1
        cluster_idxx = 1
        leng = len(points)
        
        # Find the highest points after smoothing and initialize as seed point.
        max_value = np.max(points[:,0])
        pose = np.argmax(points[:,0])
        points_index = np.array([pose])
        points[pose,1] = cluster_idxx
        
        # Find front points, which are the neighbours of already assigned points.
        # By initialization, the front points are just neighbours of seed point.
        # Check if index exceeds the length of list
        neighbours_index= neighbours_mask+pose
        neighbours_index[neighbours_index<0] = 0
        neighbours_index[neighbours_index>=leng] = leng-1
        
        # Make a boolean list to store front points.
        # Also make a list to store cluster index of front points,
        # which indicates where the front points come from.
        front = np.zeros(leng,dtype=bool)
        front[neighbours_index] = True
        front_idx = np.zeros(leng)
        front_idx[front] = cluster_idxx
        
        
        
        # Iterate until water level goes down to 0
        if stop_at == 'bottom' or stop_at == 'second_peak':
            a = 0
        else:
            a = stop_at
        while water_level >= a:
            water_level -= step_width
            
            # Make boolean list to store all active points,
            # which are the points above current water level and not yet assigned.
            # Note points has the axis (matrix_value, cluster_index)
            active_points = points[:,0] > water_level*max_value
            
            # Remove already assigned points from active points
            active_points[points[:,1] != 0] = False
            
            # Iterate until active points are all assigned
            while active_points.any():
                
                # Iterate until active points and front list have no overlapp
                new_points = np.logical_and(active_points,front)
                while new_points.any():
                    
                    # Assign active points to corresponding cluster
                    # Set cluster index of new points to front index
                    new_points_idx = np.argwhere(new_points).flatten()
                    points_index = np.concatenate((points_index, new_points_idx))
                    points[new_points,1] = front_idx[new_points]
                    
                    # Remove assigned points from active points
                    active_points[new_points] = False
                    
                    # Update front list, which are the neighbours of already assigned points
                    # Find index of neighbours.
                    new_front_index = points[new_points,1]
                    for i in range(len(neighbours_mask)):
                        if i == 0 or i == 3 or i ==5:
                            temp = new_points_idx%self.matrix_size!=0
                        elif i == 2 or i == 4 or i == 7:
                            temp = new_points_idx%self.matrix_size!=self.matrix_size-1
                        else:
                            temp = np.ones_like(new_points_idx,dtype=bool)
                        neighbours_index = new_points_idx[temp] + neighbours_mask[i]
                        
                        # Check if index exceeds the length of list
                        neighbours_index[neighbours_index<0] = 0
                        neighbours_index[neighbours_index>=leng] = leng-1 
                        
                        # Update front list
                        front[neighbours_index] = True
                        front_idx[neighbours_index] = new_front_index[temp]
                        
                    # Update new points
                    new_points = np.logical_and(active_points,front)
                    
                    
                # If active points still remain, initialize new cluster
                if active_points.any():
                    cluster_idxx += 1
                    pose = np.argmax(points[active_points,0])
                    pose = np.argwhere(active_points)[pose]
                    points_index = np.concatenate((points_index,pose))
                    points[pose,1] = cluster_idxx
                    
                    # Update front list
                    neighbours_index = neighbours_mask+pose
                    neighbours_index[neighbours_index<0] = 0
                    neighbours_index[neighbours_index>=leng] = leng-1 
                    front[neighbours_index] = True
                    front_idx[neighbours_index] = cluster_idxx
                    active_points[pose] = False
                    if stop_at == 'second_peak':
                        return water_level

            
            # Return the number of clusters and there mass
            percentage = np.array(())
            s = len(points_index)
            for i in range(cluster_idxx):
                p = np.sum(points[:,1] == i+1)/s
                percentage = np.append(percentage, p)
               
                
        # Save image if needed
        if save_image == True:
            # Convert to image
            points_2d = points[:,1].reshape(self.matrix_size,self.matrix_size)
            points_2d = np.round(255 - points_2d*255/np.max(points_2d))
            cv2.imwrite("water_level_"+str(round(water_level*100))+".png", points_2d)

        if stop_at == 'second_peak':
            return water_level
        else:
            return cluster_idxx, percentage, points[:,1].reshape(self.matrix_size,self.matrix_size)
        
    
    
        
    
    
    
    def calc_kurtosis(self, radius = 0.5, line_width = 1):
        """Calculate the kurtosis of the highest peak within range of radius
           The kortosis will only be evaluated in the main direction of the matrix (direction of street)
           """
           
        # Convert radius into matrix space
        radius = int(np.floor(radius/self.max_cons_grid_edge))
        
        # Use Gaussian Filter to smooth matrix
        matrix_filtered = cv2.GaussianBlur(self.Matrix, (7,7), 1, borderType=cv2.BORDER_REFLECT101)
        
        # Find the line (direction of street) from Max Cons Matrix using covarince matrix
        # Again find candidate points and ues there matrix value as weights
        pos = self.Matrix > 0.6*self.highest_point_value
        points = np.argwhere(pos)
        points =points[:,0:2]
        weights = matrix_filtered[points[:,0],points[:,1]]
        weights[weights<1] = 1
        
        # Use Hough Transformation to find the heading of the line 
        # Function returns the heading angle of line in radian and its din=stance to (0,0)
        theta, _ = Hough_Transformation(points, weights)
        # print(line_number)
        
        # Build rotation matrix
        rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                    [math.sin(theta), math.cos(theta)]])
        
        
        # Find all the neighbors within given radius.
        # Make a flat array of all neighbours.
        # Here we build neighbours at (0,0) firstly to find points belong to main direction,
        # then shift neighours to peak coordinate
        neighbours = np.array([np.array([x,y]) 
                               for x in range(-radius, radius + 1)
                               for y in range(-radius, radius + 1)        
                               if x**2 + y**2 <= radius**2])
        
        # Find the x,y coordinate of highest point
        [x_center,y_center] = self.highest_point_coordinate[0:2]
        
        # Rotate neighbours and find the index of line points
        neighbours_rotated = np.dot(neighbours,rotation_matrix)
        line_idx = abs(neighbours_rotated[:,1]) <= line_width
        
        # Find line points
        # Shift neighours back to peak location
        neighbours = neighbours[line_idx] + [x_center,y_center]
        neighbours_rotated = neighbours_rotated[line_idx] 
        
        
        # Find values of line points and use it to duplicate line points
        repetitions = np.array([])
        for idx in neighbours:
            [x,y] = idx
            if np.max(idx) < self.matrix_size and np.min(idx) >= 0: 
                repetitions = np.append(repetitions, matrix_filtered[x,y])
            else:
                # If index value out of range of matrix, reflect them 
                [x,y] = 2*np.array([x_center,y_center]) - idx
                if np.max(idx) < self.matrix_size and np.min(idx) >= 0: 
                    repetitions = np.append(repetitions, matrix_filtered[x,y])
                else:
                    repetitions = np.append(repetitions, 0)
        
        # Duplicate points with its value in matrix to build cooresponding distribution
        repetitions -= np.min(repetitions)
        neighbours_rotated = np.repeat(neighbours_rotated, repetitions.astype(int), axis=0)
        
        # Finally, calculate kurtosis and compare with threshold
        self.kurtosis = stats.kurtosis(neighbours_rotated[:,0])
        self.kurtosis_score = 0
        if self.kurtosis < -1:
            self.kurtosis_score = 1
        
        return self.kurtosis_score


    
    def get_matrix_score(self, peak_score=1, prec_score=1, critical_peak_threshold=0.95):
        
        if self.critical_peak_number == 0:
            self.critical_peak_score = 0
            self.score = 0
            self.prec_score = 0
            return 0
        
        # Initialize critical peak score
        # Critical peaks will be weighted by its distance to the highest peak to get score
        self.critical_peak_score = 0
        for local_peak in self.critical_peaks:
            # The colume of local peak are 
            # (x, y, peak_value, difference to highest peak, percentage in candidate points)
            if local_peak[3] >= critical_peak_threshold:
            # Calculate distance between highest peak and current local peak
            # Use distance as weight
                distance = np.linalg.norm(local_peak[0:2] - self.shift[0:2])
                self.critical_peak_score += distance
        
        self.score = 0
        self.score += self.critical_peak_number * peak_score
        self.prec_score = 0
        if self.critical_peaks[0,4] <= 1.2/self.peak_number:
            self.prec_score = prec_score
        
        #self.score += self.critical_peak_score
    
    
    
    def find_matched_points(self, map_points, scan_points, idx_map, idx_scan, indice, unique = False):
        """Find matched points in map data and scan data for compairation in Cloud Compair"""
        # Indice contains offsets between matched map points and matched scan points
        # The indice where offset equal to max cons shift are the index of points we are looking for 
        # Note one scan points may have multipel corresponding map points
        # Set unique to True to get unique set of matched points

        idx = np.logical_and.reduce(indice-self.highest_point_coordinate == 0, axis=1)
     
        # Also compare indice with (0,0) to get correctly matched points
        idx2 = np.logical_and.reduce(indice==0, axis=1)
        
        # Return founded points
        matched_map_points = map_points[idx_map][idx]
       
        matched_scan_points = scan_points[idx_scan][idx]
        correct_map_points = map_points[idx_map][idx2]
        
        # Make a unique set of points by finding the closest matched points, if applies. 
        if unique == True:
            # Find unique index of matched_scan_points
            # Note if one point occurs multipel time in scan points, 
            # np.unique will return only the index of its first occurrence
            # Sort the index from small to large
            _, unique_indices = np.unique(matched_scan_points, return_index=True, axis=0)
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
                    m_points_tmp = matched_map_points[range(ini,i)][:,0:3]
                    s_points_tmp = matched_scan_points[range(ini,i)][:,0:3]
                    closest_points_idx = np.argmin(np.linalg.norm(m_points_tmp-s_points_tmp,axis=1))
                    unique_idx_list.append(ini+closest_points_idx)
                    ini = i
 
            # Reutrn unique point set
            matched_map_points = matched_map_points[unique_idx_list]
            matched_scan_points = matched_scan_points[unique_idx_list]
            
        # Since z axis is scaled up duo to the search process, scale down z to its orginal coordinate 
        matched_map_points[:,2] /= self.z_scale
        matched_scan_points[:,2] /= self.z_scale
        correct_map_points[:,2] /= self.z_scale
        
        # Evaluate the distribution of matched points using covarience matrix
        points = matched_map_points[:,0:3]
      
        points = points - np.mean(points, axis = 0)
        cov_matrix = np.cov(points.T)
        eig_value, eig_vector = np.linalg.eig(cov_matrix)
        
        return matched_map_points, matched_scan_points, correct_map_points, eig_value
    
    
    
    def show(self):
        """Plot evaluation results"""
        print("highest peak value: " + str(self.highest_point_value))
        
        print("shift: " + str(self.shift))
        
        print("peak number: " + str(self.peak_number))
        
        print("critical peak number: " + str(self.critical_peak_number))
        
        print("critical peaks: " + str(self.critical_peaks))
        
        print("critical peak score: " + str(self.critical_peak_score))

        # print("total score: ")
        # print(self.score)
        # print('\n')