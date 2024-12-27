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
        # Подписки на топики
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)            # Данные с одометрии
        self.lidar_subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)         # Данные с лидара
        self.usage_subscription = self.create_subscription(Bool, '/lidar/use_lidar', self.usage_callback, 10)   # Использование лидара

        # Публикаторы
        self.result_publisher = self.create_publisher(String, '/comp/task_3', 10)   # Завершение работы ноды для этапа 3
        self.mode_publisher = self.create_publisher(String, '/comp/drive', 10)      # Режим отслеживания полос
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)               # Управление роботом

        # Флаги
        self.usage = False  # Использование лидара
        self.drive = True   # Контроль работы drive_node
        self.start = True   # Если доехали до первой стены - отключаем drive_node
        self.turning = False        # Робот поворачивается   
        # self.lidar_wall_3 = True    
        # self.test = False
        # self.test3 = True
        # self.test4 = True
        # self.turn_right = False     # Робот повернулся напрао
        # self.main3task = False
        # self.do_2_steni = False

        self.lidar_data = None    # Данные с лидара
        self.target_yaw = None    # Целевой угол поворота
        self.current_yaw = None    # Текущий угол поворота
        self.wall_counter = 0      # Счетчик стен
        self.turn_counter = 0      # Счетчик поворотов

    def usage_callback(self, msg):
        self.usage = msg.data
        self.get_logger().info('Lidar is in use' if msg.data else 'Lidar is NOT in use')

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
    
    def toggle_drive(self):
        msg = String()
        self.drive = not self.drive
        msg.data = "both" if self.drive else "nothing"
        self.mode_publisher.publish(msg)

        # self.get_logger().info(f"Published {msg.data}")

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def check_turn_completion(self):
        self.get_logger().info(f"{self.target_yaw}, {self.current_yaw}")
        angle_diff = self.normalize_angle(self.target_yaw - self.current_yaw)
        if abs(angle_diff) < 0.05:  # Увеличенный порог для завершения поворота
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            self.turning = False

    def turn_robot_yaw(self, target):
        target_angle = target * (math.pi / 180)  # Преобразуем градусы в радианы
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
        if self.turn_counter < 5:
            if self.lidar_data is None:
                return

            front_index = 0  # Точка спереди
            left_index = 89   # Точка слева
            right_index = 269  # Точка справа

            front_point = self.lidar_data[front_index]
            left_point = self.lidar_data[left_index]
            right_point = self.lidar_data[right_index]

            self.adjust_movement_with_lidar(front_point, left_point, right_point)

    def adjust_movement_with_lidar(self, front_point, left_point, right_point):
        # self.get_logger().info(f"{front_point}, {left_point}, {right_point}, {self.turn_counter}")
        twist = Twist()
        # Проверяем, есть ли препятствие спереди и робот не поворачивается
        if front_point < 0.28 and not self.turning and (self.turn_counter in [0, 2]):
            if self.start:
                self.start = False
                self.toggle_drive()

            self.get_logger().info(f"Поворот {self.turn_counter}")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            # self.test1 = False
            # self.main3task = True
            # self.wall_counter = 1
            # self.test = True
            self.turning = True
            self.turn_robot_yaw(87 if self.turn_counter == 0 else -87)  # Поворот налево
            self.check_turn_completion()
            self.turn_counter += 1
        
        # Если препятствие справа и условие self.test выполнено, то двигаемся вперед
        # elif right_point < 0.6 and self.test and self.main3task and not self.turning:
        elif right_point < 0.6 and self.turn_counter == 1 and not self.turning:
            self.get_logger().info("Вперед 1")
            twist.linear.x = 0.2
            twist.angular.z = 0.0
            # self.test = False
            self.publisher.publish(twist)

        # Если препятствие слева и справа свободно, то останавливаемся и поворачиваем налево
        # elif right_point > 0.6 and left_point < 0.4 and not self.test and self.wall_counter > 1 and self.test4:
        elif right_point > 0.6 and left_point < 0.4 and self.turn_counter == 1 and not self.turning:
            self.get_logger().info("Поворот направо")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)

            self.turn_counter += 1
            self.turning = True
            self.turn_robot_yaw(-84)  # Поворот направо
            self.check_turn_completion()
            # self.test4 = False
            # self.do_2_steni = True

        # Если прошли поворот и двигаемся вдоль стены
        # elif self.do_2_steni and not self.turning:
        elif self.turn_counter == 2 and not self.turning:
            self.get_logger().info("Вперед 2")
            twist.linear.x = 0.2
            self.publisher.publish(twist)
            # self.do_2_steni = False
        
        # Если препятствие спереди и робот уже повернулся один раз
        # elif front_point < 0.28 and not self.turning and self.wall_counter == 1:
        # elif front_point < 0.28 and not self.turning and self.turn_counter == 2:
        #     # self.get_logger().info("Поворот направо последний")
        #     twist4 = Twist()
        #     twist4.linear.x = 0.0
        #     twist4.angular.z = 0.0
        #     # self.test = False
        #     self.turning = True
        #     # self.turn_right = True
        #     self.publisher.publish(twist4)
        #     self.turn_counter = 3
        #     self.turn_robot_yaw(-80)  # Поворот напрвао
        #     self.check_turn_completion()

        # Если препятствие слева и робот повернулся направо
        # elif left_point < 0.6 and self.turn_right and self.test3 and not self.turning:
        # elif front_point < 0.6 and self.turn_counter == 3 and not self.turning:
        # elif self.turn_counter == 3 and not self.turning:
        #     # self.get_logger().info("Движемся на выход")         
        #     self.get_logger().info("Вперед 3")
        #     twist.linear.x = 0.12
        #     self.publisher.publish(twist)

        elif left_point < 0.6 and self.turn_counter == 3 and not self.turning:
            self.get_logger().info("Вперед 3")
            twist.linear.x = 0.2
            twist.angular.z = 0.0
            # self.test = False
            self.publisher.publish(twist)

        elif left_point > 0.6 and right_point < 0.4 and self.turn_counter == 3 and not self.turning:
            self.get_logger().info("Поворот налево")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)

            self.turn_counter += 1
            self.turning = True
            self.turn_robot_yaw(60)  # Поворот направо
            self.check_turn_completion()
        
        elif self.turn_counter == 4 and not self.turning:
            twist.linear.x = 0.10
            twist.angular.z = -0.05
            # self.test3 = False
            self.publisher.publish(twist)
            
            self.turn_counter += 1
            self.toggle_drive()

            msg = String()
            msg.data = "Done"
            self.result_publisher.publish(msg)

            rclpy.logging.get_logger("Task 3 Node").info('Task completed. Quit working...')
            raise SystemExit


        


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