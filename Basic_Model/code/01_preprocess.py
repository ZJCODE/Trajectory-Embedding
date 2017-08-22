#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:03:02 2017

@author: zhangjun
"""

import numpy as np
import glob
import time
from math import acos,sin,cos,pi
import matplotlib.pyplot as plt


def time2timestamp(t):
    timestamp = int(time.mktime(time.strptime(t,'%Y-%m-%d %H:%M:%S')))
    return timestamp


def round_num(x,r=2):
    return round(float(x),r)


def remove_stay_point(time_list,position_list):
    time_list_new = [time_list[0]]
    position_list_new = [position_list[0]]
    for t,p in zip(time_list,position_list):
        if p == position_list_new[-1]:
            pass
        else:
            time_list_new.append(t)
            position_list_new.append(p)
    return time_list_new,position_list_new

def remove_outlier(time_list,position_list,speed_threshold = 100):
    
    time_list_new = [time_list[0]]
    position_list_new = [position_list[0]]
    for t,p in zip(time_list,position_list):
        try:            
            speed = distance(p,position_list_new[-1]) / ((t-time_list_new[-1]+1)/3600.0)
            # Speed filter 
            if speed > speed_threshold :
                pass
            else:
                time_list_new.append(t)
                position_list_new.append(p)
        except:
            time_list_new.append(t)
            position_list_new.append(p)
    return time_list_new,position_list_new


def distance(p1,p2):
    from_lng,from_lat = p1
    to_lng,to_lat  = p2
    dist = 6378.137*acos(sin(to_lat/57.2958) * sin(from_lat/57.2958) + cos(to_lat/57.2958) * cos(from_lat/57.2958) * cos(to_lng/57.2958 - from_lng/57.2958))
    return dist

# segmentation
# 1. based on time interval. 
# 2. shape of a trajectory (turning points) .
# 3. based on the stay points it contains


def trajectory_segmentation(time_list,position_list,time_threshold = 3600,angel_threshold = 170,min_len = 10):
    
    # segmentation : time interval
    time_list_diff = np.diff(time_list)
    index = np.arange(len(time_list_diff))
    split_piont = (index[time_list_diff > time_threshold] + 1).tolist()
    
    
    # segmentation : shape of a trajectory (turning points)
    
    # 实时转向
    direction = np.diff(position_list,axis=0)
    direction_norm = np.linalg.norm(direction,axis=1)
    for i in range(1,len(direction)):
        if direction_norm[i] > 0:
            angle = np.dot(direction[i],direction[i-1]) / (direction_norm[i] * direction_norm[i-1])
            angle = min(1,max(-1,angle))
            angle = acos(angle)*1.0/pi*180
            if angle > angel_threshold:
                split_piont.append(i+1)

    # 大方向上的转向    
    direction_whole = (np.array(position_list) - np.array(position_list)[0])[1:]
    direction_whole_norm = np.linalg.norm(direction,axis=1)
    for i in range(1,len(direction)):
        if direction_norm[i] > 0:
            angle = np.dot(direction[i],direction_whole[i]) / (direction_norm[i] * direction_whole_norm[i])
            angle = min(1,max(-1,angle))
            angle = acos(angle)*1.0/pi*180
            if angle > angel_threshold:
                split_piont.append(i+1)


    split_piont = sorted(split_piont)
    split_piont = [0] + split_piont
    
    # get sub_trajectory
    sub_trajectory = []
    for i in range(1,len(split_piont)):
        sub_trajectory.append(position_list[split_piont[i-1]:split_piont[i]])
        
    sub_trajectory.append(position_list[split_piont[-1]:])
    sub_trajectory = [s_t for s_t in sub_trajectory if len(s_t) >= min_len]
    return sub_trajectory

def plot_trajectory(trajectory,show_step = True):
    trajectory = np.array(trajectory)
    x = trajectory[:,0]
    y = trajectory[:,1]
    plt.plot(x,y,'.-')
    if show_step :        
        index = range(len(x))
        for i_x,i_y,i in zip(x,y,index):
            plt.text(i_x,i_y,str(i))


def list2pair(X,index = False,with_start_end_tag = False):
    if with_start_end_tag:
        # add start
        if index:
            line = str(index) + '|' + 'start\t' + '-'.join(map(str,X[0]))+'\n'
        else:
            line = 'start\t' + '-'.join(map(str,X[0]))+'\n'
    else:
        line = ''

    for i in range(1,len(X)):
        if index:
            line += str(index) + '|' + '-'.join(map(str,X[i-1])) + '\t' + '-'.join(map(str,X[i])) + '\n'
        else:
            line += '-'.join(map(str,X[i-1])) + '\t' + '-'.join(map(str,X[i])) + '\n'

    if with_start_end_tag:
        # add end
        if index:
            line += str(index) + '|' +  '-'.join(map(str,X[-1])) + '\tend\n'
        else:
            line += '-'.join(map(str,X[-1])) + '\tend\n'

    return line








raw_data_path = '../data/raw_data'
raw_data_files = glob.glob(raw_data_path + '/*.txt')


speed_threshold = input('speed_threshold [100] : ')
time_threshold = input('time_threshold [3600] : ')
angel_threshold = input('angel_threshold [150] : ')
min_len = input('min_length [10] : ')
with_start_end_tag = input('add start,end tag or not ? [1 : add / 0 : not add] : ')

'''
trajectory_extract_path = '../data/trajectory' + '-[speed_threshold_' \
                            + str(speed_threshold) + '_time_threshold_' \
                            + str(time_threshold) + '_angel_threshold_' \
                            + str(angel_threshold) + '_min_len_' + str(min_len) + ']'
    
trajectory_extract_node_pair_path = '../data/trajectory_node_pair' + '-[speed_threshold_' \
                            + str(speed_threshold) + '_time_threshold_' \
                            + str(time_threshold) + '_angel_threshold_' \
                            + str(angel_threshold) + '_min_len_' + str(min_len) +']'       


trajectory_extract_node_pair_with_index_path = '../data/trajectory_node_pair_with_index' + '-[speed_threshold_' \
                            + str(speed_threshold) + '_time_threshold_' \
                            + str(time_threshold) + '_angel_threshold_' \
                            + str(angel_threshold) + '_min_len_' + str(min_len) +']'   
'''
trajectory_extract_path = '../data/trajectory'
trajectory_extract_node_pair_path = '../data/trajectory_node_pair'
trajectory_extract_node_pair_with_index_path = '../data/trajectory_node_pair_with_index'

                            
trajectory_extract = open(trajectory_extract_path,'w')
trajectory_extract_pair = open(trajectory_extract_node_pair_path,'w')
trajectory_extract_pair_with_index = open(trajectory_extract_node_pair_with_index_path,'w')


#sub_trajectory = []

trajectory_num = 0
for raw_data_file in raw_data_files:
    
    f = open(raw_data_file,'r')
    time_list = []
    position_list = []
    
    for line in f.readlines():
        num,tm,lng,lat = line.strip().split(',')
        timestamp = time2timestamp(tm)
        lng,lat = map(round_num,(lng,lat))
        time_list.append(timestamp)
        position_list.append([lng,lat]) 
    
    if len(time_list) > min_len:
               
        time_list,position_list = remove_stay_point(time_list,position_list)
        time_list,position_list = remove_outlier(time_list,position_list,speed_threshold = speed_threshold)
        sub_trajectory_part = trajectory_segmentation(time_list,position_list,time_threshold = time_threshold,angel_threshold = angel_threshold,min_len = min_len)
    
        for s_t in sub_trajectory_part:
            line = str(trajectory_num) + '|' + ','.join(map(str,s_t)) + '\n'
            trajectory_extract.write(line)
            trajectory_extract_pair.write(list2pair(s_t,index = False,with_start_end_tag = with_start_end_tag))
            trajectory_extract_pair_with_index.write(list2pair(s_t,index = str(trajectory_num),with_start_end_tag = with_start_end_tag))
            trajectory_num += 1
            if trajectory_num % 1000 == 0:
                print 'already written %d trajectorys '%(trajectory_num)
        #sub_trajectory += sub_trajectory_part
    
    f.close()

print '\n -----------\n all got %d trajectorys'%(trajectory_num)

trajectory_extract.close()
trajectory_extract_pair.close()
trajectory_extract_pair_with_index.close()




'''
plt.rc('figure',figsize=[10,8])
for i in sub_trajectory:
    plot_trajectory(i,False)
plt.legend(map(str,range(len(sub_trajectory))))
'''