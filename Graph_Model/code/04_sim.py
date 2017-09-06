import numpy as np
import glob

def cos_sim(X):
    X_v = np.array(X)
    dot = np.dot(X_v,X_v.transpose())
    n = np.linalg.norm(X_v,axis=1)
    nn = np.dot(n.reshape(-1,1),n.reshape(1,-1))
    cos_similarity = dot/nn
    return cos_similarity


def load_trajectory_vec(path):
    f = open(path,'r')
    trajectory_matrix = []
    trajectory_num_list = []
    for line in f.readlines():
        line_split = line.strip().split(' ')
        trajectory_num = int(line_split[0].split('_')[1])
        trajectory_num_list.append(trajectory_num)
        trajectory_vec = map(float,line_split[1:])
        trajectory_matrix.append(trajectory_vec)
    trajectory_matrix = np.array(trajectory_matrix)
    f.close()
    return trajectory_num_list,trajectory_matrix


trajectory_num_list,trajectory_matrix = load_trajectory_vec('../data/result')

trajectory_num_dict = dict(zip(range(len(trajectory_num_list)),trajectory_num_list))

trajectory_sim = cos_sim(trajectory_matrix)

trajectory_sim_order_index = trajectory_sim.argsort()[:,::-1]

sim_filter_threshold = input('sim_filter_threshold : ')


trajectory_sim_result_path = '../data/trajectory_sim_result'
f = open(trajectory_sim_result_path,'w')

for sim_order_index ,sim  in zip(trajectory_sim_order_index,trajectory_sim):
    sim = sim[sim_order_index]
    sim_order_index = sim_order_index[sim > sim_filter_threshold]
    sim = [round(x,3) for x in sim[sim > sim_filter_threshold]]
    zip_index_sim = zip([trajectory_num_dict[x] for x in sim_order_index],sim)
    line = str(trajectory_num_dict[sim_order_index[0]]) + '\t' + '\t'.join(map(str,zip_index_sim[1:])) + '\n'
    print line
    f.write(line)
f.close()