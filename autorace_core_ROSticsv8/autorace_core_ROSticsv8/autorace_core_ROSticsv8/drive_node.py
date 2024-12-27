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
import time
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from collections import deque


class DriveNode(Node):
    def __init__(self):
        super().__init__('drive_node')
        # Публикатор
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)   # Управление роботом
        
        # Подписки
        self.mode_subscriber = self.create_subscription(String, '/comp/drive', self.mode_callback, 10)      # Режим отслеживания полос
        self.task_subscriber = self.create_subscription(Int32, '/comp/task', self.task_callback, 10)      # Текущий этап
        self.image_subscriber = self.create_subscription(Image, '/color/image', self.image_callback, 10)    # Изображения с камер
        
        # Режим отслеживания полос, 0 - обе, 1 - желтая, 2 - белая, 3 - не отслеживать, -1 - остановиться
        self.mode = 3
        self.task = 0

        self.bridge = CvBridge()

        # Коэффициенты pid-регулятора
        self.proportional = 0.0025
        self.integral = 0.0001
        self.differential = 0.0001
        self.target = 0.3

        # Ошибки для pid-регулятора
        self.error = 0                      # Накопленная интегральная ошибка
        self.errors = [0] * 15
        self.previous_error = 0             # Предыдущая ошибка
        self.previous_time = time.time()    # Время предыдущего измерения

        # Ограничения для линейной и угловой скоростей
        self.angular_max = 0.6
        self.linear_max = 0.08
        # self.linear_max_on_turn = 0.1
        
        # Коэффициент пропорциональности линейной скорости к угловой
        self.velocity_coeff = 1.75
        
    def task_callback(self, msg):
        self.task = msg.data
        if msg.data == 3:
            meow = 1

    def mode_callback(self, msg):
        if msg.data == 'yellow':
            self.mode = 1
        elif msg.data == 'white':
            self.mode = 2
        elif msg.data == 'both':
            self.mode = 0
        elif msg.data == 'nothing':
            self.mode = -1
            self.get_logger().info("Stop moving")
        
        # self.mode = msg.data
        self.get_logger().info(f"Mode \"{self.mode}\" enabled!")

        # if msg.data == 'right':
        #     self.get_logger().info('Turn right')
        # elif msg.data == 'left':
        #     self.get_logger().info('Turn left')
        # elif msg.data == 'stop':
        #     self.get_logger().info('Stop')

    def image_callback(self, msg):
        if self.mode == -1:
            meow = 1
            # message = Twist()
            # message.linear.x = 0.0
            # message.angular.z = 0.0

            # self.publisher.publish(message)
        elif self.mode < 3:
            # Получение изображения и конвертация в HSV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Диапазоны цветов
            lower_white = np.array([0, 0, 240])
            upper_white = np.array([180, 30, 255])
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])

            # Маски объектов по цветам
            # mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
            # mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

            # mask = cv2.bitwise_or(mask_white, mask_yellow)

            height, width, _ = cv_image.shape
            bottom_third = height - height // 4

            # Определяем точки для перспективной трансформации
            src_points = np.float32([[0, bottom_third], [width, bottom_third], [0, height], [width, height]])
            dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

            # Вычисляем матрицу трансформации
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_image = cv2.warpPerspective(cv_image, M, (width, height))

            # Преобразуем трансформированное изображение в HSV
            warped_hsv_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
            warped_mask_white = cv2.inRange(warped_hsv_image, lower_white, upper_white)
            warped_mask_yellow = cv2.inRange(warped_hsv_image, lower_yellow, upper_yellow)

            # Считаем количество белых и желтых пикселей
            cnt_white = np.sum(warped_mask_white == 255)
            cnt_yellow = np.sum(warped_mask_yellow == 255)

            # Обрабатываем желтые пиксели
            right_yellow_points = []
            cx_yellow = -1
            cy_yellow = -1
            cnt_yellow_left = 0
            cnt_yellow_right = 0
            first_cx_yellow = -1
            last_cx_yellow = -1
            first_cy_yellow = -1
            last_cy_yellow = -1

            if cnt_yellow != 0:
                for y, row in enumerate(warped_mask_yellow):
                    non_zero_indices = np.nonzero(row)[0]

                    valid_indices = non_zero_indices[non_zero_indices < width//2]
                    no_valid_indices = non_zero_indices[non_zero_indices >= width//2]
                    cnt_yellow_left += len(valid_indices)
                    cnt_yellow_right += len(no_valid_indices)

                    if len(valid_indices) > 0:
                        rightmost_index = valid_indices[-1]
                        right_yellow_points.append([rightmost_index, y])

                if right_yellow_points:  # Проверяем, что массив не пуст
                    right_yellow_points = np.array(right_yellow_points)
                    centroid_yellow = np.mean(right_yellow_points, axis=0)

                    cy_yellow = int(centroid_yellow[1])
                    cx_yellow, cy_yellow = right_yellow_points[np.argmin(np.abs(right_yellow_points[:, 1] - cy_yellow))]
                    first_cx_yellow, first_cy_yellow = right_yellow_points[0]
                    last_cx_yellow, last_cy_yellow = right_yellow_points[-1]

            # Обрабатываем белые пиксели
            left_white_points = []
            cx_white = -1
            cy_white = -1
            cnt_white_left = 0
            cnt_white_right = 0
            first_cx_white = -1
            last_cx_white = -1
            first_cy_white = -1
            last_cy_white = -1

            if cnt_white != 0:
                for y, row in enumerate(warped_mask_white):
                    non_zero_indices = np.nonzero(row)[0]

                    valid_indices = non_zero_indices[non_zero_indices >= width//2]
                    no_valid_indices = non_zero_indices[non_zero_indices < width//2]
                    cnt_white_left += len(no_valid_indices)
                    cnt_white_right += len(valid_indices)

                    if len(valid_indices) > 0:
                        leftmost_index = valid_indices[0]
                        left_white_points.append([leftmost_index, y])

                if left_white_points:  # Проверяем, что массив не пуст
                    left_white_points = np.array(left_white_points)
                    centroid_white = np.mean(left_white_points, axis=0)

                    cy_white = int(centroid_white[1])
                    cx_white, cy_white = left_white_points[np.argmin(np.abs(left_white_points[:, 1] - cy_white))]
                    first_cx_white, first_cy_white = left_white_points[0]
                    last_cx_white, last_cy_white = left_white_points[-1]

            # Определяем режим движения на основе обнаруженных пикселей
            if cy_white > -1 and cy_yellow > -1:
                ava_mode = 0

                if self.task == 2:
                    if cnt_white_left / (cnt_white_right + cnt_white_left + 1) > 0.1:
                        ava_mode = 2
                    elif cnt_yellow_right / (cnt_yellow_left + cnt_yellow_right + 1) > 0.1:
                        ava_mode = 1
            elif cy_white == -1 and cy_yellow > -1:
                ava_mode = 1
            elif cy_white > -1 and cy_yellow == -1:
                ava_mode = 2
            else:
                ava_mode = 3

            # Уточняем режим движения в зависимости от текущего задания
            if self.mode == 0:
                real_mode = ava_mode
            elif self.mode == 1:
                if ava_mode == 1:
                    real_mode = ava_mode
                else:
                    real_mode = self.mode
            elif self.mode == 2:
                if ava_mode == 2:
                    real_mode = ava_mode
                else:
                    real_mode = self.mode

            # Вычисляем угол и ошибку для управления движением
            angle = 0
            tg = 0
            if real_mode == 0:
                cv2.circle(warped_image, (cx_white, cy_white), 5, (255, 0, 0), -1)
                cv2.circle(warped_image, (cx_yellow, cy_yellow), 5, (0, 0, 255), -1)
                cv2.line(warped_image, (cx_white, cy_white), (cx_yellow, cy_yellow), (0, 255, 0), 2)

                cx_center = (cx_white + cx_yellow) // 2
                cy_center = (cy_white + cy_yellow) // 2

                cx_first = int((first_cx_white + first_cx_yellow) / 2)
                cy_first = int((first_cy_white + first_cy_yellow) / 2)
                cx_last = int((last_cx_white + last_cx_yellow) / 2)
                cy_last = int((last_cy_white + last_cy_yellow) / 2)

                angle = abs((cx_first - cx_last) / (cy_first - cy_last + 1))

                cv2.circle(warped_image, (cx_center, cy_center), 5, (0, 255, 0), -1)
                cv2.imshow("Center", warped_image)
                cv2.waitKey(1)

                error = (width // 2 - cx_center)
                tg = abs((cy_white - cy_yellow) / (cx_white - cx_yellow))

            elif real_mode == 1:
                cv2.circle(warped_image, (cx_yellow + 320, 240), 5, (255, 0, 0), -1)
                cv2.circle(warped_image, (cx_yellow, cy_yellow), 5, (0, 0, 255), -1)
                cv2.line(warped_image, (cx_yellow + 320, 240), (cx_yellow, cy_yellow), (0, 255, 0), 2)

                cx_center = cx_yellow + 160
                cy_center = (cy_yellow + 240) // 2

                cx_first = int(first_cx_yellow + 160)
                cy_first = int(first_cy_yellow // 2) 
                cx_last = int(last_cx_yellow + 160)
                cy_last = int((last_cy_yellow + 480) // 2)

                angle = abs((cx_first - cx_last) / (cy_first - cy_last + 1))

                cv2.circle(warped_image, (cx_center, cy_center), 5, (0, 255, 0), -1)
                cv2.imshow("Center", warped_image)
                cv2.waitKey(1)

                error = (width // 2 - cx_center)
                tg = abs((240 - cy_yellow) / 320)

            elif real_mode == 2:
                cv2.circle(warped_image, (cx_white, cy_white), 5, (255, 0, 0), -1)
                cv2.circle(warped_image, (cx_white - 320, 240), 5, (0, 0, 255), -1)
                cv2.line(warped_image, (cx_white - 320, 240), (cx_white, cy_white), (0, 255, 0), 2)

                cx_center = cx_white - 160
                cy_center = (240 + cy_white) // 2

                cx_first = int(first_cx_white - 160)
                cy_first = int(first_cy_white // 2) 
                cx_last = int(last_cx_white - 160)
                cy_last = int((last_cy_white + 480) // 2)

                angle = abs((cx_first - cx_last) / (cy_first - cy_last + 1))

                cv2.circle(warped_image, (cx_center, cy_center), 5, (0, 255, 0), -1)
                cv2.imshow("Center", warped_image)
                cv2.waitKey(1)

                error = (width // 2 - cx_center)
                tg = abs((240 - cy_white) / 320)

            else:
                error = 0

            # Отправляем команду на движение с учетом вычисленных параметров
            # self.send_movement_command(error, tg, angle)
            self.move(error, tg, angle)
    
    # cv2.imshow("YOLOv8 Detection", cv_image)
    # cv2.waitKey(1)

    def compute_pid(self, current_error):
        current_time = time.time()
        dt = current_time - self.previous_time
        self.previous_time = current_time

        error_proportional = self.proportional * current_error
        
        self.integral += current_error * dt
        error_integral = self.integral * self.error
        
        error_differential = (current_error - self.previous_error) / dt if dt > 0 else 0
        error_differential *= self.differential
        
        self.previous_error = current_error

        return error_proportional + error_integral + error_differential
    

    def move(self, error, tg, angle):
        # Kp = self.get_parameter("Kp").get_parameter_value().double_value
        # Ki = self.get_parameter("Ki").get_parameter_value().double_value
        # Kd = self.get_parameter("Kd").get_parameter_value().double_value
        # desiredV = self.get_parameter("desiredV").get_parameter_value().double_value

        # # Вычисление ошибки
        current_error = error
        error_integral = sum(self.errors) + error
        error_differential = error - self.previous_error
        # e_P = error
        # e_I = sum(self.E) + error
        # e_D = error - self.old_e

        # Вычисление угловой скорости
        # angular_velocity = self.compute_pid(error)
        # angular_velocity = (abs(angular_velocity) / angular_velocity) * self.angular_max if angular_velocity > self.angular_max else angular_velocity
        angular_velocity = current_error * self.proportional + error_integral * self.integral + error_differential * self.differential
        angular_velocity = max(min(angular_velocity, self.angular_max), -self.angular_max)

        self.errors.pop(0)
        self.errors.append(error)
        self.previous_error = error
        
        # Динамическое управление линейной скоростью
        # Уменьшаем линейную скорость пропорционально угловой скорости
        angle_loss = 4
        angular_loss = 2
        linear_velocity = max(self.linear_max, self.target / (np.exp(angular_loss * abs(angular_velocity) / self.angular_max) * np.exp(angle_loss * angle)))

        # limit = self.linear_max if angular_velocity < (self.angular_max / 3) else self.linear_max_on_turn
        # linear_velocity = limit * max(0, 1 - self.velocity_coeff * abs(angular_velocity))
        # linear_velocity = self.linear_max * (1 - abs(angular_velocity) / self.angular_max)

        message = Twist()
        message.linear.x = linear_velocity
        message.angular.z = angular_velocity

        self.publisher.publish(message)

        # Ограничение угловой скорости
        # if w > w_max:
        #     w = w_max
        # elif w < -w_max:
        #     w = -w_max

        # Обновление стека интегральной ошибки
        # self.E.pop(0)
        # self.E.append(error)
        # self.old_e = error
        
        # Динамическое управление линейной скоростью
        # Уменьшаем линейную скорость пропорционально угловой скорости
        # angle_loss = 4
        # angular_loss = 2
        # velocity_tresh = 0.05
        # linear_velocity = max(velocity_tresh, desiredV / (np.exp(angular_loss * abs(w) / w_max) * np.exp(angle_loss * angle)))


        # Обновление команды управления
        # twist = Twist()
        # twist.linear.x = linear_velocity
        # twist.angular.z = float(w)

        # self.publisher.publish(twist)
        

# def send_finish_message(self, message_text):
#     message = String()
#     message.data = message_text
#     self.finish_publisher.publish(message)
#     self.get_logger().info(f'Сообщение "{message_text}" отправлено в топик /robot_finish.')

# def detect_objects_with_yolo(self, cv_image):
#     results = self.model(cv_image, verbose=False, conf=0.96)
#     labels = []

#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             label = result.names[int(box.cls)]
            
#             confidence = box.conf.item()  # Получаем точность распознавания
#             if x2 > 140 and x2 < 700:
#                 labels.append(label)
#             cv2.putText(cv_image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#             if label == "works_sign" and self.current_task > 1:
#                 self.lidar_3_task = True  
#             if label == "parking_sign" and confidence > 0.989 and self.temp_4_task and not self.finish:
#                 self.finish = True
#                 self.send_finish_message("ROSticsv8")

#     return cv_image, labels
        




# rclpy.logging.get_logger("Task 1 Node").info('Message was sent. Quit working...')
# raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    drive = DriveNode()
    rclpy.logging.get_logger("Drive Node").info('Working...')

    try:
        rclpy.spin(drive)
    except SystemExit:
        rclpy.logging.get_logger("Drive Node").info('Done!')
    finally:
        drive.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()