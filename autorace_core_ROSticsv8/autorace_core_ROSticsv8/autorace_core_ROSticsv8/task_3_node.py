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


class LidarNode(Node):
    def __init__(self):
        super().__init__('task_3_node')
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)        # Одометрия
        self.lidar_subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)     # Лидар

        self.lidar_subscription = self.create_subscription(Bool, '/lidar/use_lidar', self.usage_callback, 10)
        self.mode_publisher = self.create_publisher(String, '/comp/drive', 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
            # rclpy.logging.get_logger("Task 2 Node").info('Task completed. Quit working...')
            # raise SystemExit
        self.usage = False
        self.test = False
        self.start = True
        self.lidar_data = None
        self.target_yaw = None
        self.current_yaw = None
        self.turning = False
        # self.lidar_3_task = False
        self.stopped = False
        
        self.mode = 0
        self.wall_counter = 0
        self.turn_right = False
        self.test3 = True
        self.test4 = True
        self.main3task = False
        self.do_2_steni = False
        self.temp_4_task = False
        self.finish = False

    def usage_callback(self, msg):
        self.usage = msg.data
        self.get_logger().info('Lidar is in use' if msg.data else 'Lidar is NOT in use')

    def stop_drive_node(self):
        msg = String()
        msg.data = "nothing"
        self.mode_publisher.publish(msg)
        self.get_logger().info("Published -1")

    def odom_callback(self, msg):
        if self.usage:
            # Получаем кватернион из сообщения
            orientation = msg.pose.pose.orientation
            quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
            _, _, yaw = euler_from_quaternion(quaternion)
            
            # Сохраняем текущий угол
            self.current_yaw = yaw
            if self.turning:
                self.check_turn_completion()

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def check_turn_completion(self):
        angle_diff = self.normalize_angle(self.target_yaw - self.current_yaw)
        if abs(angle_diff) < 0.05:  # Увеличенный порог для завершения поворота
            self.stop_robot()
            self.turning = False

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
        self.get_logger().info("Робот остановлен.")

    def turn_robot_yaw(self, target):
        target_angle = target * math.pi / 180  # Преобразуем градусы в радианы
        self.target_yaw = self.normalize_angle(self.current_yaw + target_angle)  # Нормализуем целевой угол
        self.turning = True
        
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = math.copysign(0.4, target_angle)  # Устанавливаем скорость поворота с учетом знака
        
        self.publisher.publish(twist)

    def lidar_callback(self, msg):
        if self.usage:
            self.lidar_data = msg.ranges
            self.process_lidar_data()

    def process_lidar_data(self):
        # if self.lidar_3_task:
        if self.lidar_data is None:
            return

        front_index = 0  # Точка спереди
        left_index = 89   # Точка слева
        right_index = 269  # Точка справа

        front_point = self.lidar_data[front_index]
        left_point = self.lidar_data[left_index]
        right_point = self.lidar_data[right_index]

        # if self.lidar_3_task:
        self.adjust_movement_with_lidar(front_point, left_point, right_point)

    def adjust_movement_with_lidar(self, front_point, left_point, right_point):
        # Проверяем, есть ли препятствие спереди и робот не поворачивается
        if front_point < 0.28 and not self.turning and self.wall_counter == 0:
            if self.start:
                self.stop_drive_node()
                self.get_logger().info("Lidar started")
            
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.moving = False
            self.test1 = False
            self.main3task = True
            self.wall_counter = 1
            self.test = True
            self.turning = True
            self.publisher.publish(twist)
            self.turn_robot_yaw(87)  # Поворот на 90 градусов
            self.check_turn_completion()
        
        # Если препятствие справа и условие self.test выполнено, то двигаемся вперед
        elif right_point < 0.6 and self.test and self.main3task and not self.turning:
            twist1 = Twist()
            twist1.linear.x = 0.11
            twist1.angular.z = 0.0
            self.test = False
            self.publisher.publish(twist1)

        # Если препятствие слева и справа свободно, то останавливаемся и поворачиваем налево
        elif right_point > 0.6 and left_point < 0.4 and not self.test and self.main3task and self.test4:
            twist2 = Twist()
            twist2.linear.x = 0.0
            twist2.angular.z = 0.0
            self.publisher.publish(twist2)
            self.turning = True
            self.turn_robot_yaw(-80)  # Поворот на 90 градусов
            self.check_turn_completion()
            self.test4 = False
            self.do_2_steni = True

        # Если прошли поворот и двигаемся вдоль стены
        elif self.do_2_steni and not self.turning:
            twist3 = Twist()
            twist3.linear.x = 0.1
            self.publisher.publish(twist3)
            self.do_2_steni = False
        
        # Если препятствие спереди и робот уже повернулся один раз
        elif front_point < 0.28 and not self.turning and self.wall_counter == 1:
            twist4 = Twist()
            twist4.linear.x = 0.0
            twist4.angular.z = 0.0
            self.moving = False
            self.test = False
            self.turning = True
            self.turn_right = True
            self.publisher.publish(twist4)
            self.turn_robot_yaw(-80)  # Поворот на 90 градусов
            self.check_turn_completion()

        # Если препятствие слева и робот повернулся направо
        elif left_point < 0.6 and self.turn_right and self.test3 and not self.turning:
            twist5 = Twist()
            twist5.linear.x = 0.01
            twist5.angular.z = 0.3
            self.test3 = False
            self.publisher.publish(twist5)
            
            self.lidar_3_task = False
            self.temp_4_task = True



def main(args=None):
    rclpy.init(args=args)
    task_3 = LidarNode()
    
    rclpy.logging.get_logger("Task 3 Node").info('Working...')

    try:
        rclpy.spin(task_3)
    except SystemExit:
        rclpy.logging.get_logger("Task 3 Node").info('Done!')
    finally:
        task_3.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()