#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2

def callback(data):
    print("--- callback ---")
    print(data.width)
    print(data.height)
    print(data.point_step)
    print(data.row_step)
    print(data.fields)
    print(len(data.data))

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("velodyne_points", PointCloud2, callback)
    rospy.spin()

if __name__=='__main__':
    listener()
