cp ../data/trajectory_node_pair_for_LINE LINE
rm ../data/trajectory_node_vec_*
cd LINE
sh get_node_vec.sh $1
cp trajectory_node_vec_* ../../data
rm trajectory_node*
