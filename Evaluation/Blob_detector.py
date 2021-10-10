# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 19:03:38 2021

@author: Joachim
"""

import time
import glob
import numpy as np
from functions import Calc_blob_radius, Calc_blob_radius2, Hough_Transformation, calc_covarince_score_matrix, Hough_space_accumulator
from MaxConsMatrix import MaxConsMatrix
import h5py
import os
import open3d as o3d
from scipy.spatial import cKDTree
import sys
import cv2




if __name__ == "__main__":
    max_cons_radius = 1
    max_cons_grid_edge = 0.04
    
    max_cons_z_range = 0.05
    
    max_cons_heading_range = 5
    max_cos_grid_angle = 1

    search_model = 'xy'
    remove_ground = True
    
    
    path = 'C:/Users/Joachim/Desktop/Masterarbeit/Code/Matrix/new_data/cov_matrix/lod3_0.04_hard_radius0.04_maxnn10000/*.txt'
    files  = glob.glob(path) 
    blob_size_list = []
    RS = np.empty((1,3))
    
    step_width_ = 0.1
    sigma_max_ = 8
    
    count = 0
    for i in range(len(files)):
        # if count != 637:
        #     count += 1
        #     continue
        if search_model == 'xy':
            matrix = np.loadtxt(files[i])
        else:
            with h5py.File(files[i], 'r') as f:
                matrix = f.get("cov_matrix")[:]
        # matrix = cv2.GaussianBlur(matrix, (21,21), 1, borderType=cv2.BORDER_ISOLATED)
        matrix = np.sqrt(matrix)
        r, shift = Calc_blob_radius2(matrix, step_width = step_width_, sigma_max = sigma_max_)
        # RS = np.concatenate((RS, r[np.newaxis,:]))
        print(r)
        # print(shift)
        blob_size_list.append(r[0])
        print(count)
        count += 1
    
    # np.savetxt('blob_center.txt', RS[:,1:3]*max_cons_grid_edge-max_cons_radius)
    
    blob_size_list = np.round(np.array(blob_size_list),1)
    
    
    
    histo = []
    for j in np.round(np.arange(1, sigma_max_+step_width_,step_width_),1):
        histo.append(np.sum(blob_size_list==j))