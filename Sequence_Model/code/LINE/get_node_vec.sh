
./line -train trajectory_node_pair_for_LINE  -output trajectory_node_vec_order_1_dim_$1 -size $1 -order 1
./line -train trajectory_node_pair_for_LINE  -output trajectory_node_vec_order_2_dim_$1 -size $1 -order 2
python combine.py trajectory_node_vec_order_1_dim_$1 trajectory_node_vec_order_2_dim_$1


rm trajectory_node_vec_order_1*
rm trajectory_node_vec_order_2*