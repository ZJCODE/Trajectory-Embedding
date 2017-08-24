import tensorflow as tf 
from tensorflow.contrib.layers import fully_connected,layer_norm
from tensorflow.contrib import rnn
import numpy as np
import glob
import os

def deal_with_node_vec_line(line,normalize = False):
    line = line.strip()
    line_split = line.split(' ')
    node = line_split[0]
    vec  = np.array(map(float,line_split[1:]))
    if normalize:
        vec = (vec - vec.mean()) / vec.std()
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
    return train_data # each element in the train_data list is a sequence who's length is time_step_size


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
            trajectory_node_vec_mean_matrix.append(np.array(node_vec_sum).mean(0).tolist())
            node_vec_sum = [node_vec]
        index_pre = index
    trajectory_node_vec_mean_matrix.append(np.array(node_vec_sum).mean(0).tolist())

    trajectory_node_vec_mean_matrix = np.array(trajectory_node_vec_mean_matrix)
    return trajectory_node_vec_mean_matrix

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


# Coef
trajectory_embedding_size = node_vec_size
learning_rate = input('learning_rate : ') #0.1
batch_size = input('batch_size : ') #128 
time_step_size =  input('time_step_size : ')
skip_step = input('skip_step (how many steps to skip before reporting the loss) : ')  # how many steps to skip before reporting the loss
loss_report_times = input('loss_report_times : ')
num_train_step = skip_step * loss_report_times
combine_type = input('combine_type [ 0 : concat / 1 : element_wise_multiply ] : ')
combine_type_dict = {0:'concat',1:'element_wise_multiply'}

def trajectory_embedding_seq_model(batch_gen):

    with tf.variable_scope("input"):

        trajectory_index = tf.placeholder(tf.int32, shape=[batch_size,time_step_size],name = 'trajectory_index')
        node_vec_seq = tf.placeholder(tf.float32, shape=[batch_size,time_step_size,node_vec_size],name = 'node_inputs')

    with tf.variable_scope("embedding"):
        trajectory_embedding = tf.Variable(tf.random_uniform([trajectory_num, trajectory_embedding_size], -1.0, 1.0),name = 'trajectory_embedding')
        trajectory_index_embed = tf.nn.embedding_lookup(trajectory_embedding, trajectory_index)



    with tf.variable_scope("combine"):

        if combine_type == 0:
            try:
                node_trajectory_vec = tf.concat([node_vec_seq,trajectory_index_embed],2,name = 'node_trajectory_vec')
            except:
                node_trajectory_vec = tf.concat(2,[node_vec_seq,trajectory_index_embed],name = 'node_trajectory_vec')
        elif combine_type == 1:
            node_trajectory_vec = tf.multiply(node_vec_seq,trajectory_index_embed,name = 'node_trajectory_vec')


    node_vec_seq_T = tf.transpose(node_vec_seq,[1,0,2])
    node_vec_seq_R = tf.reshape(node_vec_seq_T, [-1,node_vec_size])
    node_vec_seq_split = tf.split(node_vec_seq_R, time_step_size, 0)


    # node_trajectory_vec  : (batch_size, time_step_size, vec_size)
    vec_size = int(node_trajectory_vec.get_shape()[2])
    lstm_input_T = tf.transpose(node_trajectory_vec,[1,0,2])
    lstm_input_R = tf.reshape(lstm_input_T, [-1,vec_size])
    # lstm_input : [(batch_size, vec_size),(batch_size, vec_size)...] , length is time_step_size
    lstm_input = tf.split(lstm_input_R, time_step_size, 0)
    
    lstm_size = node_vec_size

    #------- drietion : go ahead -------

    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    # lstm_output : [(batch_size, lstm_size),(batch_size, lstm_size)...] , length is time_step_size
    lstm_output, _states = rnn.static_rnn(lstm, lstm_input, dtype=tf.float32)
    
    ##------- drietion : go back -------

    node_vec_seq_split_inverse = node_vec_seq_split[::-1]
    lstm_input_inverse = lstm_input[::-1]
    lstm_output_inverse, _states_inverse = rnn.static_rnn(lstm, lstm_input_inverse, dtype=tf.float32)

    # pos1 + index_pos1 -> result  | min dist(result , pos2)
    loss = tf.reduce_mean(tf.square(tf.subtract(node_vec_seq_split[1:],lstm_output[:-1]))) + tf.reduce_mean(tf.square(tf.subtract(node_vec_seq_split_inverse[1:],lstm_output_inverse[:-1])))
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

            # set trajectory_node_vec_mean_matrix still during the pretrain of lstm's coef
            if num < num_train_step/2:
                sess.run(tf.assign(trajectory_embedding, trajectory_node_vec_mean_matrix))
            index_batch,node_vec_batch = next(batch_gen)
            loss_batch, _ = sess.run([loss, optimizer], 
                                        feed_dict={trajectory_index: index_batch, node_vec_seq:node_vec_batch})
            total_loss += loss_batch

            if (num + 1) % skip_step == 0:
                loss_report_round += 1
                print('Average loss at loss_report_round {}: {:5.5f}'.format(loss_report_round, total_loss * 1.0 / skip_step))
                print sess.run(trajectory_embedding)[0,:]
                loss_list.append(total_loss * 1.0 / skip_step)
                total_loss = 0.0

        trajectory_embedding_matrix = sess.run(trajectory_embedding)

    return trajectory_embedding_matrix,loss_list



def main():

    batch_gen = generate_batch_data(train_data_path,node_vec_path,batch_size,time_step_size)

    trajectory_embedding_matrix,loss_list = trajectory_embedding_seq_model(batch_gen)
    trajectory_embedding_matrix_path = '../data/trajectory_embedding_matrix' \
                                        + '_time_step_size_' +str(time_step_size) \
                                        + '_embedding_size_' + str(trajectory_embedding_size) \
                                        + '_learning_rate_' + str(learning_rate) \
                                        + '_batch_size_' + str(batch_size) \
                                        + '_num_train_step_' + str(num_train_step) \
                                        + '_node_vec_size_' + str(node_vec_size) \
                                        + '_combine_type_' + combine_type_dict[combine_type] \
                                        + '.npy'
    loss_list_path = '../data/loss_list' \
                    + '_time_step_size_' +str(time_step_size) \
                    + '_embedding_size_' + str(trajectory_embedding_size) \
                    + '_learning_rate_' + str(learning_rate) \
                    + '_batch_size_' + str(batch_size) \
                    + '_num_train_step_' + str(num_train_step) \
                    + '_node_vec_size_' + str(node_vec_size) \
                    + '_combine_type_' + combine_type_dict[combine_type] \
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







