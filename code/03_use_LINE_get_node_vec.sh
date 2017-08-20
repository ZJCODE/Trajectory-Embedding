cp ../data/trajectory_node_pair_for_LINE LINE
rm ../data/trajectory_node_vec_*
cd LINE
sh get_node_vec.sh
cp trajectory_node_vec_order_* ../../data
rm trajectory_node*
