./line -train trajectory_node_pair_for_LINE  -output trajectory_node_vec_order_2_dim_$1 -size $1 -order 2
sed -i '1d' trajectory_node_vec_order_2_dim_$1
