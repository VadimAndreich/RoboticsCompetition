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


class CrossroadNode(Node):
    def __init__(self):
        super().__init__('task_2_node')
        # Подписки на топики
        self.label_subscriber = self.create_subscription(String, '/yolo/label', self.label_callback, 10)    # Полученный с yolo лейбл
        self.publisher = self.create_publisher(String, '/comp/task_2', 10)      # Завершение работы ноды
        
        # Публикаторы
        self.task_publisher = self.create_publisher(Int32, '/comp/task', 10)     # Этап соревнования
        self.mode_publisher = self.create_publisher(String, '/comp/drive', 10)   # Режим отслеживания полос
    
    
    def label_callback(self, msg):
        task = Int32()
        mode = String()
        
        # Для поворота направо отслеживаем белую линию
        if msg.data == 'right':
            task.data = 2
            self.task_publisher.publish(task)

            mode.data = 'white'
            self.mode_publisher.publish(mode)

            self.get_logger().info('Turn right')
        # Для поворота налево отслеживаем желтую линию
        elif msg.data == 'left':
            task.data = 2
            self.task_publisher.publish(task)

            mode.data = 'yellow'
            self.mode_publisher.publish(mode)

            self.get_logger().info('Turn left')
        # После потери знака из вида отслеживаем обе линии
        elif msg.data == 'lost':
            mode.data = 'both'
            self.mode_publisher.publish(mode)
            self.get_logger().info('Both')
        # После обнаружения знака дорожных работ отслеживаем белую линию
        elif msg.data == 'work':
            task.data = 3
            self.task_publisher.publish(task)

            mode.data = 'white'
            self.mode_publisher.publish(mode)

            self.get_logger().info('Preparing for task 3')

            # Завершаем работу ноды для этапа 2
            result = String()
            result.data = 'Done'
            self.publisher.publish(result)
            
            rclpy.logging.get_logger("Task 2 Node").info('Task completed. Quit working...')
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    task_2 = CrossroadNode()
    
    rclpy.logging.get_logger("Task 2 Node").info('Working...')

    try:
        rclpy.spin(task_2)
    except SystemExit:
        rclpy.logging.get_logger("Task 2 Node").info('Done!')
    finally:
        task_2.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()