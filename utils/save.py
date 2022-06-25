# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 22:06:51 2022

@author: Joachim
"""
import numpy as np
import h5py
from functions import  Hough_space_accumulator

def save_restuls_to_local(accumulator, search_model, count):
   
    if search_model == 'xy':
        np.savetxt(str(count).zfill(5)+".txt", accumulator)
        
    else:
        with h5py.File(str(count).zfill(5)+".hdf5", 'w') as f:
            f.create_dataset("accumulator", data=accumulator, compression="gzip", compression_opts=5)
    
    
def save_hough_matrix(Accumulator, points_0, points_1, matched_points_idx, matrix_size, heading_search_size, search_model, count):
    
    hough_matrix = Hough_space_accumulator(points_0, points_1, matched_points_idx, matrix_size, heading_search_size, search_model)
    
    if search_model == 'xy':
        a,b = Accumulator.highest_point_coordinate
        np.savetxt(str(count).zfill(5)+".txt", hough_matrix[a,b])

    else:
        a,b,c = Accumulator.highest_point_coordinate
        np.savetxt(str(count).zfill(5)+".txt", hough_matrix[a,b,c])
        