#!/usr/bin/env python

import rospy
import pyqtgraph as pg
import sys
import pickle
from std_msgs.msg import Float32MultiArray, Float32
from PyQt5.QtGui import *
from PyQt5.QtCore import *
class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("Result")
        self.setGeometry(50, 50, 1210, 650)
        self.graph_sub = rospy.Subscriber('result', Float32MultiArray, self.data)
        self.ep = []
        self.data = []
        self.qvalue = []
        self.rewards = []
        self.actorloss = []
        self.criticloss = []
        self.x = []
        self.count = 1
        self.size_ep = 0
        load_data = True

        if load_data:
            self.ep, self.data = self.load_data()
            for index in self.data:
                self.rewards.append(index[0])
                self.qvalue.append(index[1])
                self.actorloss.append(index[2])
                self.criticloss.append(index[3])
            self.size_ep = len(self.ep)

        self.plot()

    def data(self, data):
        self.ep.append(self.size_ep + self.count)
        self.count += 1
        self.rewards.append(data.data[0])
        self.qvalue.append(data.data[1])
        self.actorloss.append(data.data[2])
        self.criticloss.append(data.data[3])
        self.data.append([ self.rewards[-1],  self.qvalue[-1],  self.actorloss[-1],  self.criticloss[-1]])

    def plot(self):
        self.rewardsPlt = pg.PlotWidget(self, title="Total reward")
        self.rewardsPlt.move(0, 10)
        self.rewardsPlt.resize(600, 300)

        self.timer1 = pg.QtCore.QTimer()
        self.timer1.timeout.connect(self.update)
        self.timer1.start(200)

        self.qValuePlt = pg.PlotWidget(self, title="Median Q-value")
        self.qValuePlt.move(0, 320)
        self.qValuePlt.resize(600, 300)

        self.timer2 = pg.QtCore.QTimer()
        self.timer2.timeout.connect(self.update)
        self.timer2.start(200)

        self.actor_loss_Plt = pg.PlotWidget(self, title="Loss of actor")
        self.actor_loss_Plt.move(610, 10)
        self.actor_loss_Plt.resize(600, 300)

        self.timer3 = pg.QtCore.QTimer()
        self.timer3.timeout.connect(self.update)
        self.timer3.start(200)

        self.critic_loss_Plt = pg.PlotWidget(self, title="Loss of critic")
        self.critic_loss_Plt.move(610, 320)
        self.critic_loss_Plt.resize(600, 300)

        self.timer3 = pg.QtCore.QTimer()
        self.timer3.timeout.connect(self.update)
        self.timer3.start(200)
        self.show()

    def update(self):
        self.rewardsPlt.showGrid(x=True, y=True)
        self.qValuePlt.showGrid(x=True, y=True)
        self.actor_loss_Plt.showGrid(x=True, y=True)
        self.critic_loss_Plt.showGrid(x=True, y=True)

        self.save_data([self.ep, self.data])

        self.rewardsPlt.plot(self.ep, self.rewards, pen=(255, 0, 0))
        self.qValuePlt.plot(self.ep, self.qvalue, pen=(0, 255, 0))
        self.actor_loss_Plt.plot(self.ep, self.actorloss, pen=(255, 255, 0))
        self.critic_loss_Plt.plot(self.ep, self.criticloss, pen=(0, 255, 255))

    def load_data(self):
        try:
            with open("graph.txt") as f:
                x, y = pickle.load(f)
        except:
            x, y = [], []
        return x, y

    def save_data(self, data):
        with open("graph.txt", "wb") as f:
            pickle.dump(data, f)


def run():
        rospy.init_node('graph')
        app = QApplication(sys.argv)
        GUI = Window()
        sys.exit(app.exec_())

run()
