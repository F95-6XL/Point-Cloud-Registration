# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 03:35:38 2021

@author: Joachim
"""
import time
import glob
import numpy as np
from functions import draw_histogram, find_outlier, Mean_Shift, Hough_Transformation
from MaxConsMatrix import MaxConsMatrix
import h5py
import os
import sys

def Read_matrix(matrix, max_cons_radius, max_cons_grid_edge, max_cons_z_range, max_cos_grid_angle, 
                max_cons_heading_range, search_model = 'xy'):
    
    z_scale = max_cons_radius / max_cons_z_range
    matrix_radius = int(np.ceil(max_cons_radius / max_cons_grid_edge))
    if 2*matrix_radius+1 != matrix.shape[0]:
        print( "Error: Wrong max_cons_radius")
        sys.exit()
    matrix_size = 2*matrix_radius+1
    heading_search_radius = int(np.ceil(max_cons_heading_range / max_cos_grid_angle))
    
    Matrix = MaxConsMatrix(matrix, max_cons_radius, max_cons_grid_edge, 
                           matrix_size, max_cons_heading_range, max_cos_grid_angle, 
                           heading_search_radius, search_model, z_scale)
    
    Matrix.calc_shift()
    Matrix.find_critical_points(p=0.9)
    a = time.time()
    water_level = Matrix.Watershed(sigma=1,step_width=0.01,stop_at ='second_peak')
    b = time.time()
    Matrix.find_peaks(critical_peak_threshold = 0.95)
    c = time.time()
    # print(b-a)
    # print(c-b)
    if search_model == 'xy':
        Matrix.calc_kurtosis(radius = 0.9, line_width=1)
    Matrix.get_matrix_score(critical_peak_threshold=0.9)
    # Matrix.show()
    
    # print(Max_Cons_Matrix.critical_peaks)
    # print(Max_Cons_Matrix.shift)
    # print(Max_Cons_Matrix.kurtosis)
    
    return Matrix, water_level

def get_score(Matrix, search_model):
    if search_model == 'xy':
        return Matrix.critical_peak_score, Matrix.score, Matrix.prec_score, Matrix.kurtosis#_score
    else:
        return Matrix.critical_peak_score, Matrix.score, Matrix.prec_score
if __name__ == "__main__":
    
    time1_begin = time.time()
    
    path1 = 'C:/Users/Joachim/Desktop/Masterarbeit/Code/Matrix/cov_matrix/lod3_0.04/*.txt' 
    path2 = 'C:/Users/Joachim/Desktop/Masterarbeit/Code/Matrix/new_data/max_cons_matrix/lod4_0.04/*.txt'
    path3 = 'C:/Users/Joachim/Desktop/Masterarbeit/Code/Matrix/new_data/cov_matrix/lod2_0.04/*.txt'
    path4 = 'C:/Users/Joachim/Desktop/Masterarbeit/Code/Matrix/new_data/cov_matrix/lod3_0.04_hard_radius0.04_maxnn10000/*.txt'
    path5 = 'C:/Users/Joachim/Desktop/Masterarbeit/Code/Matrix/new_data/cov_matrix/lod3_0.04_downsampled_voxel0.1/*.txt'
    path6 = 'C:/Users/Joachim/Desktop/Masterarbeit/Code/Matrix/new_data/max_cons_matrix/lod3_0.04_hard_radius0.04_maxnn10000/*.txt'
    path7 = 'C:/Users/Joachim/Desktop/Masterarbeit/Code/Matrix/new_data/max_cons_matrix/lod3_0.1/*.txt'
    path8 = 'C:/Users/Joachim/Desktop/Masterarbeit/Code/Matrix/new_data/cov_matrix/lod3_0.04_hard_radius0.08_maxnn100000/*.txt'
    path9 = 'C:/Users/Joachim/Desktop/Masterarbeit/Code/Matrix/old_data/lod3_0.08/*.txt'
    path_sigma = 'C:/Users/Joachim/Desktop/Masterarbeit/Code/Matrix/old_data/6_sigma/*.txt'
    
    files1 = sorted(glob.glob(path1),key=os.path.getmtime)
    files2 = glob.glob(path2) 
    files3 = glob.glob(path3) 
    files4 = glob.glob(path4) 
    files5 = glob.glob(path5) 
    files6 = glob.glob(path6) 
    files7 = glob.glob(path7) 
    files8 = glob.glob(path8) 
    files9 = sorted(glob.glob(path9),key=os.path.getmtime)
    files_sigma = glob.glob(path_sigma) 
    
    Path1 = True    # lod2_0.04
    Path2 = False    # lod3_0.04
    Path3 = False    # lod2_0.04_no_vector
    Path4 = False    # lod2_0.06
    Path5 = False    # lod2_0.04_remove_ground
    Path6 = False    # lod2_0.04_no_vegetation
    Path7 = False
    Path8 = False
    
    show = False
    # Radius of max cons search. Result will be from -radius to +radius.
    max_cons_radius = 1

    # The grid size of the max cons grid.
    max_cons_grid_edge = 0.04

    # In this implementation, z is aligned already. When aligning x/y,
    # we will search only for corresponting points in a range +/- zrange.
    max_cons_z_range = 0.05
    
    # Angle and grid size of max cons search if search of heading applies
    max_cons_heading_range = 1
    max_cos_grid_angle = 0.25
    

    search_model = 'xy'
    
    # The colume of peaks are 
    # (x, y, peak_value, difference to highest peak, percentage in candidate points, sacn index)
    peaks = np.empty((0,6))
    if search_model == 'xy':
        max_cons_shift = np.empty((0, 3))
    elif search_model == 'xyzheading':
        max_cons_shift = np.empty((0, 5))
    else:
        max_cons_shift = np.empty((0, 4))
    
    time3_begin_array = np.array([])
    time3_end_array = np.array([])
    
    ### Main ###
    water_level = []
    kurt=[]
    if Path1:   
        count = 0
        if search_model == 'xy':
            Path1_score = np.empty((0,4))
        else:
            Path1_score = np.empty((0,3))
            
            
        for name in files4:
            # if count != 1:
            #     count +=1
            #     continue
            if search_model == 'xy':
                matrix = np.loadtxt(name)
            else:
                with h5py.File(name, 'r') as f:
                    matrix = f.get("matrix")[:]
            matrix = np.sqrt(matrix)
            # matrix = matrix[10:31,10:31]
            time3_begin = time.time()
            matrix1, water = Read_matrix(matrix, max_cons_radius, max_cons_grid_edge, max_cons_z_range, 
                                max_cos_grid_angle, max_cons_heading_range, search_model)
            # peak = np.concatenate((matrix1.critical_peaks,count*np.ones(len(matrix1.critical_peaks))[:,np.newaxis]),axis=1)
            # peaks = np.concatenate((peaks, peak), axis=0)
            Path1_score = np.concatenate((Path1_score, np.array(get_score(matrix1, search_model))[np.newaxis,:]),axis=0)
            # print(water)
            water_level.append(water)
            kurt.append(matrix1.kurtosis)
            time3_end = time.time()
            
            time3_begin_array = np.append(time3_begin_array, [time3_begin])
            time3_end_array = np.append(time3_end_array, [time3_end])
            
            max_cons_shift = np.concatenate((max_cons_shift,matrix1.shift[np.newaxis,:]),axis=0)

            print(count)
            count += 1
            
        # np.savetxt('shift_lod3_0.04_downsample0.1.txt', max_cons_shift)
        # np.savetxt('peaks_lod2_0.04.txt', peaks)
        time3 = np.sum(time3_end_array - time3_begin_array) / count
        time1_end = time.time()
        time1 = time1_end - time1_begin
        
        print('time1: ' + str(time1) + '\n' + 'time3 = ' + str(time3))
    if Path2:
        Path2_score = np.empty((0,4))
        for name in files2:   
            matrix = np.loadtxt(name)
            matrix2 = Read_matrix(matrix, max_cons_radius, max_cons_grid_edge, max_cons_z_range, 
                                max_cos_grid_angle, max_cons_heading_range, search_model) 
            Path2_score = np.concatenate((Path2_score, np.array(get_score(matrix2, search_model))[np.newaxis,:]),axis=0)
    else: Path2_score=0
    
    if Path3:
        Path3_score = np.empty((0,4))
        for name in files3:
            matrix = np.loadtxt(name)   
            matrix3 = Read_matrix(matrix, max_cons_radius, max_cons_grid_edge, max_cons_z_range, 
                                max_cos_grid_angle, max_cons_heading_range, search_model) 
            Path3_score = np.concatenate((Path3_score, np.array(get_score(matrix3, search_model))[np.newaxis,:]),axis=0)
    else: Path3_score=0
        
    if Path4:
        Path4_score = np.empty((0,4))
        for name in files4:   
            matrix = np.loadtxt(name)
            matrix4 = Read_matrix(matrix, max_cons_radius, 0.06, max_cons_z_range, 
                                max_cos_grid_angle, max_cons_heading_range, search_model) 
            Path4_score = np.concatenate((Path4_score, np.array(get_score(matrix4, search_model))[np.newaxis,:]),axis=0)
    else: Path4_score=0      
    
    if Path5:
        Path5_score = np.empty((0,4))
        for name in files5:   
            matrix = np.loadtxt(name)
            matrix5 = Read_matrix(matrix, max_cons_radius, max_cons_grid_edge, max_cons_z_range, 
                                max_cos_grid_angle, max_cons_heading_range, search_model) 
            Path5_score = np.concatenate((Path5_score, np.array(get_score(matrix5, search_model))[np.newaxis,:]),axis=0)
    else: Path5_score=0
     
    if Path6:
        Path6_score = np.empty((0,4))
        for name in files6:   
            matrix = np.loadtxt(name)
            matrix6 = Read_matrix(matrix, max_cons_radius, max_cons_grid_edge, max_cons_z_range, 
                                max_cos_grid_angle, max_cons_heading_range, search_model) 
            Path6_score = np.concatenate((Path6_score, np.array(get_score(matrix6, search_model))[np.newaxis,:]),axis=0)
    else: Path6_score=0
    
    if Path7:
        Path7_score = np.empty((0,3))
        for name in files7:   
            with h5py.File(name, 'r') as f:
                matrix = f.get("matrix")[:]
            matrix7 = Read_matrix(matrix, max_cons_radius, max_cons_grid_edge, 2, 
                                max_cos_grid_angle, max_cons_heading_range, search_model='xyz') 
            Path7_score = np.concatenate((Path7_score, np.array(get_score(matrix7, search_model))[np.newaxis,:]),axis=0)
    else: Path7_score=0
    
    if Path8:
        Path8_score = np.empty((0,3))
        for name in files8:   
            with h5py.File(name, 'r') as f:
                matrix = f.get("matrix")[:]
            matrix8 = Read_matrix(matrix, max_cons_radius, max_cons_grid_edge, max_cons_z_range, 
                                max_cos_grid_angle, max_cons_heading_range, search_model='xyheading') 
            Path8_score = np.concatenate((Path8_score, np.array(get_score(matrix8, search_model))[np.newaxis,:]),axis=0)
    else: Path8_score=0
    
    if show:
        print('lod2_0.04')
        print(np.sum(Path1_score,axis=0))
        print('lod3_0.04')
        print(np.sum(Path2_score,axis=0))
        print('lod3_0.04_no_vector')
        print(np.sum(Path3_score,axis=0))
        print('lod2_0.06_grid_size')
        print(np.sum(Path4_score,axis=0))
        print('lod2_0.04_remove_ground')
        print(np.sum(Path5_score,axis=0))
        print('lod2_0.04_no_vegetation')
        print(np.sum(Path6_score,axis=0))
        print('lod2_0.04_xyz')
        print(np.sum(Path7_score,axis=0))
        print('lod2_0.04_xyheading')
        print(np.sum(Path8_score,axis=0))
water_level = np.array(water_level)
kurt = np.array(kurt)
# water_level = np.around(water_level,2)
# aaa = []
# for i in range(101):
#     aaa.append(np.sum(water_level==(100-i)/100))
aaa = draw_histogram(water_level, 1, 0, 0.05, decimal=2)
    
bbb = draw_histogram(kurt, -1, 5, 0.5, decimal=1)
# bbb = []
# for i in range(13):
#     c = (-10+i*5)/10
#     c1 = (c < kurt_)[:,np.newaxis]
#     c2 = (kurt_ < (c+0.5))[:,np.newaxis]
#     cc = np.concatenate((c1,c2),axis=1)
#     bbb.append(np.sum(np.logical_and.reduce(cc,axis=1)))