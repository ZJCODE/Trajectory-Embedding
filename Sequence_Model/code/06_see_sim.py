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

242 18307 18251
27,22052
12082,28165,5725,19710
31544,1944
31549,12409
13,29159
31533,26801,10488,9412,13733,21376,22580,28080,26360,6362,30186

'''


sim_base_f = open('../data/trajectory_sim_result_baseline_node_vec_size_16_top_5','r')
sim_embedding_f = open('../data/trajectory_sim_result_time_step_size_10_embedding_size_16_learning_rate_0.01_batch_size_512_num_train_step_100000_node_vec_size_16_combine_type_element_wise_multiply_top_5','r')

def deal_sim_line(line):
    split_line = line.strip().split('\t')
    trajectory_num = eval(split_line[0])
    if len(split_line) > 1:
        sim_trajectory_num_list = [eval(x)[0] for x in split_line[1:]]
        sim_v_trajectory_num_list = [eval(x)[1] for x in split_line[1:]]
    else:
        sim_trajectory_num_list = []
        sim_v_trajectory_num_list = []
        
    sim_trajectory_num_list = [trajectory_num] + sim_trajectory_num_list
    
    return sim_trajectory_num_list,sim_v_trajectory_num_list


sim_base_matrix = []
sim_base_matrix_v = []
for line in sim_base_f.readlines():
    n,v = deal_sim_line(line)
    sim_base_matrix.append(n)
    sim_base_matrix_v.append(v)

sim_embedding_matrix = []
sim_embedding_matrix_v = []
for line in sim_embedding_f.readlines():
    n,v = deal_sim_line(line)
    sim_embedding_matrix.append(n)
    sim_embedding_matrix_v.append(v)

459

def plot_look(n):
    node_list = sim_base_matrix[n]
    node_list_sim_v = sim_base_matrix_v[n]
    plot_trajectory(eval(trajectorys[node_list[0]]))
    print len(eval(trajectorys[node_list[0]]))
    print '----------'
    plt.legend('1')
    for node,sim_v in zip(node_list[1:],node_list_sim_v):    
        plot_trajectory(eval(trajectorys[node]))
        print len(eval(trajectorys[node])) , sim_v
    
    plt.figure()
    print '----------'
    
    node_list = sim_embedding_matrix[n]
    node_list_sim_v = sim_embedding_matrix_v[n]
    plot_trajectory(eval(trajectorys[node_list[0]]))
    plt.legend('1')
    for node,sim_v in zip(node_list[1:],node_list_sim_v):    
        plot_trajectory(eval(trajectorys[node]))
        print len(eval(trajectorys[node])) ,sim_v

