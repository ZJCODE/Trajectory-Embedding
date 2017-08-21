#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:24:48 2017

@author: zhangjun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(trajectory,show_step = True):
    trajectory = np.array(trajectory)
    x = trajectory[:,0]
    y = trajectory[:,1]
    plt.plot(x,y,'.-')
    if show_step :        
        index = range(len(x))
        for i_x,i_y,i in zip(x,y,index):
            plt.text(i_x,i_y,str(i))
            

trajectory_extract_path = '../data/trajectory'

data = pd.read_csv(trajectory_extract_path,sep='|',names= ['trajectory_index','trajectory'])

trajectorys = data.trajectory

'''
40 7837
32 15205
6 14622
242 18307 18251
 27,22052
 85,2024
'''



node_list =[85,2024]
for node in node_list:    
    plot_trajectory(eval(trajectorys[node]))


