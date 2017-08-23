import tensorflow as tf 
from tensorflow.contrib.layers import fully_connected,layer_norm
import numpy as np
import glob
import os

def deal_with_node_vec_line(line,normalize = False):
    line = line.strip()
    line_split = line.split(' ')
    node = line_split[0]
    vec  = np.array(map(float,line_split[1:]))
    if normalize:
        vec = vec - vec.min() / (vec.max() - vec.min())
    return node,vec.tolist()

def load_node_vec_dict(node_vec_path,skip_line = False,normalize=False):
    f = open(node_vec_path,'r')
    node_vec_dict = {}
    if skip_line:
        f.readline()
    for line in f.readlines():
        node,vec = deal_with_node_vec_line(line,normalize)
        node_vec_dict[node] = vec
    return node_vec_dict

def load_train_data(train_data_path,time_step_size = 10):
    '''
    train_data_path data = [a,b,c,d,e,f] 
    time_step_size = 3   -> [a,b,c] [b,c,d] [c,d,e] [d,e,f]
    '''
    train_data = []
    f = open(train_data_path)
    i = 0
    step_list = []
    for line in f.readlines():
        i += 1
        if i < time_step_size:
            step_list.append(line.strip())
        elif i == time_step_size:
            step_list = step_list+[line.strip()]
            train_data.append(step_list)
        else:
            step_list = step_list[1:]+[line.strip()]
            train_data.append(step_list)
    return train_data


def deal_with_train_data_line(line,node_vec_dict):
    step_size_index = []
    step_size_node_vec = []
    for x in line:
        node,index = x.split(',')
        index = int(index)
        node_vec = node_vec_dict[node]
        step_size_index.append(index)
        step_size_node_vec.append(node_vec)
    return step_size_index,step_size_node_vec



def generate_sample(train_data,node_vec_dict):
    while True:    
        for line in train_data:
            yield deal_with_train_data_line(line,node_vec_dict)


def get_batch(iterator,batch_size):
    while True:
        index_batch = []
        node_vec_batch = []
        for i in range(batch_size):
            index,node_vec = next(iterator)
            index_batch.append(index)
            node_vec_batch.append(node_vec)
        index_batch = np.array(index_batch)
        node_vec_batch = np.array(node_vec_batch)
        yield index_batch,node_vec_batch

def generate_batch_data(train_data_path,node_vec_path,batch_size,time_step_size):
    train_data = load_train_data(train_data_path,time_step_size)
    node_vec_dict = load_node_vec_dict(node_vec_path,skip_line = False,normalize=True)
    iterator = generate_sample(train_data,node_vec_dict)
    return get_batch(iterator,batch_size)


def node_vec_mean_of_trajectory(train_data_path,node_vec_dict):
    trajectory_node_vec_mean_matrix = []
    node_vec_sum = [] 
    index_pre = '0'
    f = open(train_data_path,'r')
    for line in f.readlines():
        node,index = line.strip().split(',')
        node_vec = node_vec_dict[node]
        if index == index_pre:
            node_vec_sum.append(node_vec)
        else:
            node_vec_sum = [node_vec]
            trajectory_node_vec_mean_matrix.append(np.array(node_vec_sum).mean(0).tolist())
        index_pre = index
    trajectory_node_vec_mean_matrix.append(np.array(node_vec_sum).mean(0).tolist())

    trajectory_node_vec_mean_matrix = np.array(trajectory_node_vec_mean_matrix)
    return trajectory_node_vec_mean_matrix

# To be modified :


# train_data
train_data_path = '../data/trajectory_sequence'

# node vector
node_vec_path_list = glob.glob('../data/trajectory_node_vec_order_combine_dim*')
zip_path = zip(range(len(node_vec_path_list)),node_vec_path_list)
print 'we have those node vec files : '
for x in zip_path:
    print x
num = input('choose which one to use [Enter the num] : ')
node_vec_path = node_vec_path_list[num]
node_vec_size =  int(node_vec_path.split('dim_')[1])*2
print 'node vector dim is : %d'%(node_vec_size)


# get num of trajectory and train data pairs 
trajectory_num = int(os.popen('wc -l ../data/trajectory').readline().split(' ')[0])
print 'all %d trajectorys '%(trajectory_num) 
data_pair_num = int(os.popen('wc -l ../data/trajectory_node_pair_with_index').readline().split(' ')[0])
print 'all %d train data '%(data_pair_num)

# Coef
# trajectory_embedding_size = input('trajectory_embedding_size : ') #32  # Dimension of the embedding vector.
trajectory_embedding_size = node_vec_size

learning_rate = input('learning_rate : ') #0.1
batch_size = input('batch_size : ') #128
print ' [ skip_step x loss_report_times = %d ] means see all data once'%(data_pair_num/batch_size) 
skip_step = input('skip_step (how many steps to skip before reporting the loss) : ')  # how many steps to skip before reporting the loss
loss_report_times = input('loss_report_times : ')
num_train_step = skip_step * loss_report_times





def trajectory_embedding_seq_model(lstm_size,batch_gen):

    pass


#  Model

def trajectory_embedding_model(batch_gen):

    with tf.variable_scope("input"):

        trajectory_index = tf.placeholder(tf.int32, shape=[batch_size],name = 'trajectory_index')
        node_vec_input = tf.placeholder(tf.float32, shape=[batch_size,node_vec_size],name = 'node_inputs')
        node_predict = tf.placeholder(tf.float32, shape=[batch_size, node_vec_size],name = 'node_predict')

    with tf.variable_scope("embedding"):
        trajectory_embedding = tf.Variable(tf.random_uniform([trajectory_num, trajectory_embedding_size], -1.0, 1.0),name = 'trajectory_embedding')
        trajectory_index_embed = tf.nn.embedding_lookup(trajectory_embedding, trajectory_index)

    with tf.variable_scope("concat"):
        try:
            node_trajectory_vec = tf.concat([node_vec_input,trajectory_index_embed],1,name = 'node_trajectory_vec')
        except:
            node_trajectory_vec = tf.concat(1 ,[node_vec_input,trajectory_index_embed],name = 'node_trajectory_vec')


    with tf.variable_scope("nn"):
        fc1 = fully_connected(node_trajectory_vec , num_outputs = (trajectory_embedding_size + node_vec_size))
        fc1_norm = layer_norm(fc1)
        fc2 = fully_connected(fc1_norm , num_outputs = (trajectory_embedding_size + node_vec_size) / 2)
        fc2_norm = layer_norm(fc2)
        result = fully_connected(fc2_norm , num_outputs = node_vec_size)

    #loss = tf.nn.l2_loss(node_predict - result,name = 'l2_loss')
    loss = tf.reduce_mean(tf.square(node_predict-result))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        node_vec_dict = load_node_vec_dict(node_vec_path,skip_line = False,normalize=True)
        trajectory_node_vec_mean_matrix = node_vec_mean_of_trajectory(train_data_path,node_vec_dict)

        # initial trajectory_embedding
        sess.run(tf.assign(trajectory_embedding, trajectory_node_vec_mean_matrix))

        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        loss_report_round = 0
        loss_list =[]
        for num in range(num_train_step):
            index,node_from_vec,node_to_vec = next(batch_gen)
            loss_batch, _ = sess.run([loss, optimizer], 
                                        feed_dict={trajectory_index: index, node_vec_input:node_from_vec, node_predict:node_to_vec})
            total_loss += loss_batch

            if (num + 1) % skip_step == 0:
                loss_report_round += 1
                print('Average loss at loss_report_round {}: {:5.5f}'.format(loss_report_round, total_loss * 1.0 / skip_step))
                loss_list.append(total_loss * 1.0 / skip_step)
                total_loss = 0.0

        trajectory_embedding_matrix = sess.run(trajectory_embedding)

    return trajectory_embedding_matrix,loss_list


def main():
    batch_gen = generate_batch_data(train_data_path,node_vec_path,batch_size)
    trajectory_embedding_matrix,loss_list = trajectory_embedding_model(batch_gen)
    trajectory_embedding_matrix_path = '../data/trajectory_embedding_matrix' \
                                        + '_embedding_size_' + str(trajectory_embedding_size) \
                                        + '_learning_rate_' + str(learning_rate) \
                                        + '_batch_size_' + str(batch_size) \
                                        + '_num_train_step_' + str(num_train_step) \
                                        + '_node_vec_size_' + str(node_vec_size) \
                                        + '.npy'
    loss_list_path = '../data/loss_list' \
                    + '_embedding_size_' + str(trajectory_embedding_size) \
                    + '_learning_rate_' + str(learning_rate) \
                    + '_batch_size_' + str(batch_size) \
                    + '_num_train_step_' + str(num_train_step) \
                    + '_node_vec_size_' + str(node_vec_size) \
                    + '.npy'
    np.save(trajectory_embedding_matrix_path,trajectory_embedding_matrix)
    np.save(loss_list_path,loss_list)

    trajectory_node_vec_mean_matrix_path = '../data/trajectory_embedding_matrix_baseline' \
                                            + '_node_vec_size_' + str(node_vec_size) \
                                            + '.npy'
    node_vec_dict = load_node_vec_dict(node_vec_path,skip_line = False)
    trajectory_node_vec_mean_matrix = node_vec_mean_of_trajectory(train_data_path,node_vec_dict)
    np.save(trajectory_node_vec_mean_matrix_path,trajectory_node_vec_mean_matrix)

if __name__ == '__main__':
    main()







