#!/usr/bin/env python

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
world = False
if world:
    from respawnGoal_custom_worlds import Respawn
else:
    from respawnGoal import Respawn
import copy
target_not_movable = False

ACTION_V_MAX = 0.3 # m/s
ACTION_W_MAX = 2. # rad/s

class Env():
    def __init__(self, action_dim=2):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.past_distance = 0.
        self.stopped = 0
        self.action_dim = action_dim
        self.v_var_rate = 0
        self.w_var_rate = 0
        self.var_rate_limit = 0.1
        self.time_step = 0.2
        #Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        #you can stop turtlebot by publishing an empty Twist
        #message
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):
        self.past_position = copy.deepcopy(self.position)
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        #print 'yaw', yaw
        #print 'gA', goal_angle

        heading = goal_angle - yaw
        #print 'heading', heading
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 3)

    def getState(self, scan, past_action):
        scan_range = []
        heading = self.heading
        min_range = 0.15
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] == float('inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] == float('nan'):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        a, c = float('{0:.3f}'.format(self.position.x)), float('{0:.3f}'.format(self.past_position.y))
        if not self.stopped:
            self.b, self.d = float('{0:.3f}'.format(self.past_position.x)),  float('{0:.3f}'.format(self.past_position.y))
        if abs(a - self.b)<=0.001 and abs(c -self.b)<=0.001:
            # rospy.loginfo('\n<<<<<Stopped>>>>>\n')
            # print('\n' + str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + '\n')
            self.stopped += 1
            if self.stopped == 20:
                rospy.loginfo('Robot is in the same 20 times in a row')
                self.stopped = 0
                done = True
        else:
            # rospy.loginfo('\n>>>>> not stopped>>>>>\n')
            self.stopped = 0

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)*15
        
        if min_range > min(scan_range) > 0:
            done = True

        # for pa in past_action:
        #     scan_range.append(pa)

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        # current_distance = self.getGoalDistace()
        if current_distance <= 0.1:
            self.get_goalbox = True
            self.stopped = 0
            
        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle],  done

    def setReward(self, state, done, action):
        current_distance = state[-3]
        heading = state[-4]
        #print('cur:', current_distance, self.past_distance)

        if done:
            rospy.loginfo("Fail!!")
            # reward = -500.
            reward = -150.
            self.pub_cmd_vel.publish(Twist())

        elif self.get_goalbox:
            rospy.loginfo("Goal!!")
            # reward = 500.
            reward = 200.
            self.pub_cmd_vel.publish(Twist())
            if world:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True, running=True)
                if target_not_movable:
                    self.reset()
            else:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        else:
            angle = heading + (pi / (ACTION_W_MAX*4) *(-action[1])) +pi
            if angle >= 0:
                angle = angle%(2*pi)
            else:
                angle = -angle
            yaw_reward = round(1* ( - math.fabs(-2/ pi * angle +2.)), 2)
            action_coef = round(2**round(-action[0]/ACTION_V_MAX, 2), 2) 
            v_var_puni = self.v_var_rate - (self.var_rate_limit * ACTION_V_MAX)
            w_var_puni = self.w_var_rate - (self.var_rate_limit * ACTION_W_MAX)
            action_var_coef = 1
            if v_var_puni > 0 and w_var_puni > 0:
                action_var_coef += (v_var_puni + w_var_puni)/2
            elif v_var_puni > 0 or w_var_puni > 0:
                if  v_var_puni > 0:
                    action_var_coef += v_var_puni
                else:
                    action_var_coef += w_var_puni
            else:
                    action_var_coef = 1

            distance_coef = 2 ** (current_distance / self.goal_distance)
            reward =0.5*action_var_coef*action_coef * yaw_reward * distance_coef

        return reward

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        self.v_var_rate = abs(past_action[0] - linear_vel)/self.time_step
        self.w_var_rate = abs(past_action[1] - ang_vel)/self.time_step

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done,  = self.getState(data, past_action)
        reward = self.setReward(state, done, past_action )

        return np.asarray(state), reward, done

    def reset(self):
        #print('aqui2_____________---')
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
        else:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)

        self.goal_distance = self.getGoalDistace()
        state, _ = self.getState(data, [0]*self.action_dim)

        return np.asarray(state)
