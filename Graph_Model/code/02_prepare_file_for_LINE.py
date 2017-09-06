#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: zhangjun
"""
from collections import Counter

#X = ['a','b','c','d','e','f','g','h']
def get_n_gram(X,size):
    X_n_gram = []
    X_length = len(X)
    for i in range(X_length-size+1):
        X_n_gram.append('#'.join(X[i:i+size]))
    return X_n_gram

def get_several_n_gram(X,size_list):
    n_gram_list = []
    for s in size_list:
        n_gram_list.append(get_n_gram(X,s))
    return n_gram_list

pair_list = []

f = open('../data/trajectory','r')

for line in f.readlines():
    #line = '0|[116.41, 39.92],[116.41, 39.93],[116.41, 39.94],[116.4, 39.94],[116.4, 39.96],[116.4, 39.98],[116.4, 39.99],[116.4, 40.08],[116.41, 40.09],[116.4, 40.13],[116.4, 40.17],[116.38, 40.21]'
    u = line.split('|')[0]
    trajectory = 'trajectory_' + u
    X = [str(x[0])+'-'+str(x[1]) for x in eval(line.split('|')[1])]
    n_gram_list = get_several_n_gram(X,size_list=[1,2,3,4,5,6])
    for n_gram in n_gram_list:
        for i in range(len(n_gram)-1):
            pair_list.append(n_gram[i]+'\t'+n_gram[i+1])
            pair_list.append(n_gram[i]+'\t'+trajectory)
        pair_list.append(n_gram[-1]+'\t'+trajectory)
f.close()


pair_count = Counter(pair_list).most_common()

file_for_LINE = '../data/trajectory_node_pair_for_LINE'

f = open(file_for_LINE,'w')

for x in pair_count:
    line = x[0] + '\t' + str(x[1]) + '\n'
    f.write(line)
f.close()
