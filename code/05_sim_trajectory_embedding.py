import numpy as np
import glob

def cos_sim(X):
    X_v = np.array(X)
    dot = np.dot(X_v,X_v.transpose())
    n = np.linalg.norm(X_v,axis=1)
    nn = np.dot(n.reshape(-1,1),n.reshape(1,-1))
    cos_similarity = dot/nn
    return cos_similarity

trajectory_embedding_matrix_path = glob.glob('../data/trajectory_embedding_matrix*')

print 'we have those embedding_matrix '

zip_path = zip(range(len(trajectory_embedding_matrix_path)),trajectory_embedding_matrix_path)

for x in zip_path:
    print x

num = input('choose which one to use [Enter the num] : ')

path = trajectory_embedding_matrix_path[num]

desc = path.split('matrix')[1].split('.npy')[0]

trajectory_embedding_matrix = np.load(path)

print 'calculate similarity matrix'
trajectory_sim = cos_sim(trajectory_embedding_matrix)
print 'sort similarity matrix'
trajectory_sim_order_index = trajectory_sim.argsort()[:,::-1]


trajectory_sim_result_path = '../data/trajectory_sim_result' + desc

sim_filter_threshold = input('sim_filter_threshold : ') #0.9

f = open(trajectory_sim_result_path,'w')

for sim_order_index ,sim  in zip(trajectory_sim_order_index,trajectory_sim):
    sim = sim[sim_order_index]
    sim_order_index = sim_order_index[sim > sim_filter_threshold]
    sim = [round(x,3) for x in sim[sim > sim_filter_threshold]]
    zip_index_sim = zip(sim_order_index,sim)
    line = str(sim_order_index[0]) + '\t' + '\t'.join(map(str,zip_index_sim[1:])) + '\n'
    print line
    f.write(line)
f.close()

