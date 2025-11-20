I comandi importanti per avviare la simulazione sono : 

ros2 run turtlebot3_perception landmark_simulator

ros2 launch turtlebot3_ignition lab04.launch.py

ros2 launch turtlebot3_bringup rviz2.launch.py

ros2 run turtlebot3_teleop teleop_keyboard

ros2 run lab03_pkg ekf_node

ros2 topic echo /ekf

ros2 bag record -a

#Questo dice al nodo: usa il clock della simulazione invece del clock reale.

ros2 run lab03_pkg ekf_node2 --ros-args -p use_sim_time:=true
