from collections import Counter
node_pair_path = '../data/trajectory_node_pair'

print 'process file : '
print node_pair_path

node_set = set()

pair_list = []
with open(node_pair_path,'r') as f:
     for line in f.readlines():
         pair_list.append(line.strip())
         f,t = line.strip().split('\t')
         node_set.add(f)
         node_set.add(t)

node_num = len(node_set)
print 'has node : %d'%(node_num)
pair_count = Counter(pair_list).most_common()

file_for_LINE = node_pair_path + '_for_LINE'

f = open(file_for_LINE,'w')

for x in pair_count:
    line = x[0] + '\t' + str(x[1]) + '\n'
    f.write(line)
f.close()

