## Base Idea

- https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning
- https://github.com/dranaju/project

## Libraries

[Pytorch]

## ROS 
You can find the packages the I used here:
- https://github.com/ROBOTIS-GIT/turtlebot3
- https://github.com/ROBOTIS-GIT/turtlebot3_msgs
- https://github.com/ROBOTIS-GIT/turtlebot3_simulations

```
cd ~/catkin_ws/src/
git clone {link_git}
cd ~/catkin_ws && catkin_make
```

To install my package you will do the same from above.

## Set State

In: turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro.

```
<xacro:arg name="laser_visual" default="false"/>   # Visualization of LDS. If you want to see LDS, set to `true`
```
And
```
<scan>
  <horizontal>
    <samples>360</samples>            # The number of sample. Modify it to 24
    <resolution>1</resolution>
    <min_angle>0.0</min_angle>
    <max_angle>6.28319</max_angle>
  </horizontal>
</scan>
```

## Run Code
I have four stage as in the examples of Robotis. But I dont know yet my code dont have a geat performance in stage 3.

First to run:
```
roslaunch turtlebot3_gazebo turtlebot3_stage_{number_of_stage}.launch
```
In another terminal run:
```
roslaunch DDPG_NAV ddpg_stage_{number_of_stage}.launch
```
If you warn test the model with actual turtlebot, first you have to make a map (see https://emanual.robotis.com/docs/en/platform/turtlebot3/slam/),and save it in folder “DDPG_NAV/maps”.
Next change the arg “map_file” in file "turtlebot3_actual_test.launch" into your map's name .
```
  <arg name="map_file" default="$(find model_test)/maps/lab.yaml"/>
```
Now link to the turtlebot and bringup(see https://emanual.robotis.com/docs/en/platform/turtlebot3/bringup/#bringup),then run:
```
roslaunch DDPG_NAV turtlebot3_actual_test.launch
```
Then use "2D Nav Goal" button to set navigation goal for DDPG(see https://emanual.robotis.com/docs/en/platform/turtlebot3/navigation/#set-navigation-goal). 


