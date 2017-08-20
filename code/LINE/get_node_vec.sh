./line -train trajectory_node_pair_for_LINE  -output trajectory_node_vec_order_1_dim_64 -size 64 -order 1
./line -train trajectory_node_pair_for_LINE  -output trajectory_node_vec_order_2_dim_64 -size 64 -order 2
python combine.py trajectory_node_vec_order_1_dim_64 trajectory_node_vec_order_2_dim_64

./line -train trajectory_node_pair_for_LINE  -output trajectory_node_vec_order_1_dim_32 -size 32 -order 1
./line -train trajectory_node_pair_for_LINE  -output trajectory_node_vec_order_2_dim_32 -size 32 -order 2
python combine.py trajectory_node_vec_order_1_dim_32 trajectory_node_vec_order_2_dim_32


./line -train trajectory_node_pair_for_LINE  -output trajectory_node_vec_order_1_dim_16 -size 16 -order 1
./line -train trajectory_node_pair_for_LINE  -output trajectory_node_vec_order_2_dim_16 -size 16 -order 2
python combine.py trajectory_node_vec_order_1_dim_16 trajectory_node_vec_order_2_dim_16


./line -train trajectory_node_pair_for_LINE  -output trajectory_node_vec_order_1_dim_8 -size 8 -order 1
./line -train trajectory_node_pair_for_LINE  -output trajectory_node_vec_order_2_dim_8 -size 8 -order 2
python combine.py trajectory_node_vec_order_1_dim_8 trajectory_node_vec_order_2_dim_8


rm trajectory_node_vec_order_1*
rm trajectory_node_vec_order_2*