import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool, Int32
import cv2
import os
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from collections import deque


class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        # Подписки на ноды для прохождения этапов
        self.start_subscription = self.create_subscription(String, '/comp/task_1', self.task_1_callback, 10)    # Старт робота по зеленому свету
        self.cross_subscription = self.create_subscription(String, '/comp/task_2', self.task_2_callback, 10)    # Выбор стороны для прохождения перекрестка 
        self.lidar_subscription = self.create_subscription(String, '/comp/task_3', self.task_3_callback, 10)    # Использование лидара для обнаружения стен
        self.finish_subscription = self.create_subscription(String, '/robot_finish', self.finish_callback, 10)  # Финиш
        
        # Публикация команд управления
        self.mode_publisher = self.create_publisher(String, '/comp/drive', 10)         # Режим отслеживания полос
        self.task_publisher = self.create_publisher(Int32, '/comp/task', 10)           # Текущий этап
        self.yolo_publisher = self.create_publisher(Bool, '/yolo/use_yolo', 10)        # Управление YOLO
        self.lidar_publisher = self.create_publisher(Bool, '/lidar/use_lidar', 10)     # Управление лидаром

        self.current_task = 1
        self.use_yolo = False
        
    def task_1_callback(self, msg):
        self.current_task = 2
        self.toggle_yolo()
        message = String()
        message.data = 'both'
        self.mode_publisher.publish(message)
        self.get_logger().info('Start.')

    def task_2_callback(self, msg):
        self.current_task = 3
        self.toggle_yolo()
        message = Bool()
        message.data = True
        self.lidar_publisher.publish(message)

    def task_3_callback(self, msg):
        self.current_task = 4
        self.toggle_yolo()

    def finish_callback(self, msg):
        message = String()
        message.data = "finish"
        self.mode_publisher.publish(message)

    def toggle_yolo(self):
        self.use_yolo = not self.use_yolo
        msg = Bool()
        msg.data = self.use_yolo
        self.yolo_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = RobotControlNode()

    rclpy.spin(image_subscriber)
    
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()