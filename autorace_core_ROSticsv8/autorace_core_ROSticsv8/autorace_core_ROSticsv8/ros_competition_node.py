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
        
        # Подписки на топики
        # self.subscription = self.create_subscription(Image, '/color/image', self.image_callback, 10)            # Изображение с камеры
        # self.yolo_subscriber = self.create_subscription(String, '/yolo/label', self.yolo_callback, 10)           # Лейблы от yolo

        # Подписки на ноды для прохождения этапов
        self.start_subscription = self.create_subscription(String, '/comp/task_1', self.task_1_callback, 10)    # Старт робота по зеленому свету
        self.cross_subscription = self.create_subscription(String, '/comp/task_2', self.task_2_callback, 10)    # Выбор стороны для прохождения перекрестка 
        self.lidar_subscription = self.create_subscription(String, '/comp/task_3', self.task_3_callback, 10)    # 
        
        # Публикация команд управления
        # self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)               # Управление роботом    
        self.mode_publisher = self.create_publisher(String, '/comp/drive', 10)         # Режим отслеживания полос
        self.task_publisher = self.create_publisher(Int32, '/comp/task', 10)          # Текущий этап
        self.finish_publisher = self.create_publisher(String, '/robot_finish', 10)  # Финиш
        self.yolo_publisher = self.create_publisher(Bool, '/yolo/use_yolo', 10)     # Управление YOLO
        self.lidar_publisher = self.create_publisher(Bool, '/lidar/use_lidar', 10)

        
        self.bridge = CvBridge()
        self.current_task = 1
        self.use_yolo = False

        self.green_detected = False
        self.moving = False
        self.test = False
        self.test1 = True
        self.test2 = False
        self.target_yaw = None
        self.turning = False
        self.lidar_3_task = False
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
        
        # Загрузка модели YOLO
        # package_name = 'robot_comp_test'
        # package_share_dir = get_package_share_directory(package_name)
        # weights_path = os.path.join(package_share_dir, 'weights', 'ros.pt')
        # self.model = YOLO(weights_path)

        # Переменные для хранения данных лидара
        # self.lidar_data = None

        # PID-регулятор
        # self.declare_parameters(
        #     namespace='',
        #     parameters=[
        #         ('Kp', 0.0025),
        #         ('Ki', 0.0001),
        #         ('Kd', 0.000),
        #         ('desiredV', 0.25),
        #     ]
        # )
        # self.len_stack = 15
        # self.E = [0] * self.len_stack
        # self.old_e = 0

    def task_1_callback(self, msg):
        self.moving = True
        self.current_task = 2
        self.toggle_yolo()
        message = String()
        message.data = 'both'
        self.mode_publisher.publish(message)
        self.get_logger().info('Начинаем движение.')

    def task_2_callback(self, msg):
        self.current_task = 3
        self.toggle_yolo()
        message = Bool()
        message.data = True
        self.lidar_publisher.publish(message)
        
        # self.get_logger().info('Начинаем движение.')

    def task_3_callback(self, msg):
        meow = 1

    def toggle_yolo(self):
        self.use_yolo = not self.use_yolo
        msg = Bool()
        msg.data = self.use_yolo
        self.yolo_publisher.publish(msg)

    # def image_callback(self, msg):
    #     cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
    #     if self.use_yolo:
    #         self.process_image(cv_image)
    #     else:

    #         cv2.imshow("Camera", cv_image)
    #         cv2.waitKey(1)

    # def yolo_callback(self, msg):


    # def odom_callback(self, msg):
    #     # Получаем кватернион из сообщения
    #     orientation = msg.pose.pose.orientation
    #     quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    #     _, _, yaw = euler_from_quaternion(quaternion)
        
    #     # Сохраняем текущий угол
    #     self.current_yaw = yaw
    #     if self.turning:
    #         self.check_turn_completion()

    # def normalize_angle(self, angle):
    #     return math.atan2(math.sin(angle), math.cos(angle))

    # def check_turn_completion(self):
    #     angle_diff = self.normalize_angle(self.target_yaw - self.current_yaw)
    #     if abs(angle_diff) < 0.05:  # Увеличенный порог для завершения поворота
    #         self.stop_robot()
    #         self.turning = False

    # def stop_robot(self):
    #     twist = Twist()
    #     twist.linear.x = 0.0
    #     twist.angular.z = 0.0
    #     self.publisher.publish(twist)
    #     self.get_logger().info("Робот остановлен.")

    # def turn_robot_yaw(self, target):
    #     target_angle = target * math.pi / 180  # Преобразуем градусы в радианы
    #     self.target_yaw = self.normalize_angle(self.current_yaw + target_angle)  # Нормализуем целевой угол
    #     self.turning = True
        
    #     twist = Twist()
    #     twist.linear.x = 0.0
    #     twist.angular.z = math.copysign(0.4, target_angle)  # Устанавливаем скорость поворота с учетом знака
        
    #     self.publisher.publish(twist)

    # def lidar_callback(self, msg):
    #     self.lidar_data = msg.ranges
    #     self.process_lidar_data()

    # def process_lidar_data(self):
    #     if self.lidar_3_task:
    #         if self.lidar_data is None:
    #             return

    #         front_index = 0  # Точка спереди
    #         left_index = 89   # Точка слева
    #         right_index = 269  # Точка справа

    #         front_point = self.lidar_data[front_index]
    #         left_point = self.lidar_data[left_index]
    #         right_point = self.lidar_data[right_index]

    #         if self.lidar_3_task:
    #             self.adjust_movement_with_lidar(front_point, left_point, right_point)

    # def adjust_movement_with_lidar(self, front_point, left_point, right_point):
    #     # Проверяем, есть ли препятствие спереди и робот не поворачивается
    #     if front_point < 0.28 and not self.turning and self.wall_counter == 0:
    #         twist = Twist()
    #         twist.linear.x = 0.0
    #         twist.angular.z = 0.0
    #         self.moving = False
    #         self.test1 = False
    #         self.main3task = True
    #         self.wall_counter = 1
    #         self.test = True
    #         self.turning = True
    #         self.publisher.publish(twist)
    #         self.turn_robot_yaw(87)  # Поворот на 90 градусов
    #         self.check_turn_completion()
        
    #     # Если препятствие справа и условие self.test выполнено, то двигаемся вперед
    #     elif right_point < 0.6 and self.test and self.main3task and not self.turning:
    #         twist1 = Twist()
    #         twist1.linear.x = 0.11
    #         twist1.angular.z = 0.0
    #         self.test = False
    #         self.publisher.publish(twist1)

    #     # Если препятствие слева и справа свободно, то останавливаемся и поворачиваем налево
    #     elif right_point > 0.6 and left_point < 0.4 and not self.test and self.main3task and self.test4:
    #         twist2 = Twist()
    #         twist2.linear.x = 0.0
    #         twist2.angular.z = 0.0
    #         self.publisher.publish(twist2)
    #         self.turning = True
    #         self.turn_robot_yaw(-80)  # Поворот на 90 градусов
    #         self.check_turn_completion()
    #         self.test4 = False
    #         self.do_2_steni = True

    #     # Если прошли поворот и двигаемся вдоль стены
    #     elif self.do_2_steni and not self.turning:
    #         twist3 = Twist()
    #         twist3.linear.x = 0.1
    #         self.publisher.publish(twist3)
    #         self.do_2_steni = False
        
    #     # Если препятствие спереди и робот уже повернулся один раз
    #     elif front_point < 0.28 and not self.turning and self.wall_counter == 1:
    #         twist4 = Twist()
    #         twist4.linear.x = 0.0
    #         twist4.angular.z = 0.0
    #         self.moving = False
    #         self.test = False
    #         self.turning = True
    #         self.turn_right = True
    #         self.publisher.publish(twist4)
    #         self.turn_robot_yaw(-70)  # Поворот на 90 градусов
    #         self.check_turn_completion()

    #     # Если препятствие слева и робот повернулся направо
    #     elif left_point < 0.6 and self.turn_right and self.test3 and not self.turning:
    #         twist5 = Twist()
    #         twist5.linear.x = 0.01
    #         twist5.angular.z = 0.3
    #         self.test3 = False
    #         self.publisher.publish(twist5)
            
    #         self.lidar_3_task = False
    #         self.temp_4_task = True

            # Можно выставить флаг, что мы прошли 3 задание

        

    # def process_image(self, cv_image):
    #     hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    #     # Диапазоны цветов
    #     lower_white = np.array([0, 0, 240])
    #     upper_white = np.array([180, 30, 255])
    #     lower_yellow = np.array([20, 100, 100])
    #     upper_yellow = np.array([30, 255, 255])

    #     # Маски объектов по цветам
    #     mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
    #     mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # cv2.imshow("Yellow", mask_yellow)
        # cv2.waitKey(1)

        # cv2.imshow("White", mask_white)
        # cv2.waitKey(1)
        
        # Если достигнут финиш, останавливаем робота
        # if self.finish:
        #     self.moving = False
        #     twist = Twist()
        #     twist.linear.x = 0.0
        #     twist.angular.z = 0.0
        #     self.publisher.publish(twist)
        # if self.temp_4_task:
        #     self.moving = True

        # Детекция через YOLO
        # if self.current_task == 2:
        #     meow = 1
            # cv_image, labels = self.detect_objects_with_yolo(cv_image)

            # # Сохраняем обнаруженные метки в список
            # self.last_labels.append(labels)

            # # Преобразуем список меток в плоский список
            # flat_list = [item for sublist in self.last_labels for item in sublist]

            # # Обрабатываем обнаруженные знаки
            # if "right_sign" in flat_list and self.current_task == 2:
            #     self.current_task = 2
            #     self.mode = 2
            # elif "left_sign" in flat_list and self.current_task == 2:
            #     self.current_task = 2
            #     self.mode = 1
            # elif not ("right_sign" in flat_list or "left_sign" in flat_list) and self.current_task == 2:
            #     self.mode = 0
            # elif self.lidar_3_task == True and self.current_task == 3:
            #     self.current_task = 3
            # elif "parking_sign" in flat_list and self.current_task > 3 and self.lidar_3_task == False:
            #     self.current_task = 4
            #     self.mode = 1

        # Если робот движется, обрабатываем маски белых и желтых объектов
        # if self.moving:
    #         mask = cv2.bitwise_or(mask_white, mask_yellow)

    #         height, width, _ = cv_image.shape
    #         bottom_third = height - height // 4

    #         # Определяем точки для перспективной трансформации
    #         src_points = np.float32([[0, bottom_third], [width, bottom_third], [0, height], [width, height]])
    #         dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    #         # Вычисляем матрицу трансформации
    #         M = cv2.getPerspectiveTransform(src_points, dst_points)
    #         warped_image = cv2.warpPerspective(cv_image, M, (width, height))

    #         # Преобразуем трансформированное изображение в HSV
    #         warped_hsv_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
    #         warped_mask_white = cv2.inRange(warped_hsv_image, lower_white, upper_white)
    #         warped_mask_yellow = cv2.inRange(warped_hsv_image, lower_yellow, upper_yellow)

    #         # Считаем количество белых и желтых пикселей
    #         cnt_white = np.sum(warped_mask_white == 255)
    #         cnt_yellow = np.sum(warped_mask_yellow == 255)

    #         # Обрабатываем желтые пиксели
    #         right_yellow_points = []
    #         cx_yellow = -1
    #         cy_yellow = -1
    #         cnt_yellow_left = 0
    #         cnt_yellow_right = 0
    #         first_cx_yellow = -1
    #         last_cx_yellow = -1
    #         first_cy_yellow = -1
    #         last_cy_yellow = -1

    #         if cnt_yellow != 0:
    #             for y, row in enumerate(warped_mask_yellow):
    #                 non_zero_indices = np.nonzero(row)[0]

    #                 valid_indices = non_zero_indices[non_zero_indices < 424]
    #                 no_valid_indices = non_zero_indices[non_zero_indices >= 424]
    #                 cnt_yellow_left += len(valid_indices)
    #                 cnt_yellow_right += len(no_valid_indices)

    #                 if len(valid_indices) > 0:
    #                     rightmost_index = valid_indices[-1]
    #                     right_yellow_points.append([rightmost_index, y])

    #             if right_yellow_points:  # Проверяем, что массив не пуст
    #                 right_yellow_points = np.array(right_yellow_points)
    #                 centroid_yellow = np.mean(right_yellow_points, axis=0)

    #                 cy_yellow = int(centroid_yellow[1])
    #                 cx_yellow, cy_yellow = right_yellow_points[np.argmin(np.abs(right_yellow_points[:, 1] - cy_yellow))]
    #                 first_cx_yellow, first_cy_yellow = right_yellow_points[0]
    #                 last_cx_yellow, last_cy_yellow = right_yellow_points[-1]

    #         # Обрабатываем белые пиксели
    #         left_white_points = []
    #         cx_white = -1
    #         cy_white = -1
    #         cnt_white_left = 0
    #         cnt_white_right = 0
    #         first_cx_white = -1
    #         last_cx_white = -1
    #         first_cy_white = -1
    #         last_cy_white = -1

    #         if cnt_white != 0:
    #             for y, row in enumerate(warped_mask_white):
    #                 non_zero_indices = np.nonzero(row)[0]

    #                 valid_indices = non_zero_indices[non_zero_indices >= 424]
    #                 no_valid_indices = non_zero_indices[non_zero_indices < 424]
    #                 cnt_white_left += len(no_valid_indices)
    #                 cnt_white_right += len(valid_indices)

    #                 if len(valid_indices) > 0:
    #                     leftmost_index = valid_indices[0]
    #                     left_white_points.append([leftmost_index, y])

    #             if left_white_points:  # Проверяем, что массив не пуст
    #                 left_white_points = np.array(left_white_points)
    #                 centroid_white = np.mean(left_white_points, axis=0)

    #                 cy_white = int(centroid_white[1])
    #                 cx_white, cy_white = left_white_points[np.argmin(np.abs(left_white_points[:, 1] - cy_white))]
    #                 first_cx_white, first_cy_white = left_white_points[0]
    #                 last_cx_white, last_cy_white = left_white_points[-1]

    #         # Определяем режим движения на основе обнаруженных пикселей
    #         if cy_white > -1 and cy_yellow > -1:
    #             ava_mode = 0

    #             if self.current_task == 2:
    #                 if cnt_white_left / (cnt_white_right + cnt_white_left + 1) > 0.1:
    #                     ava_mode = 2
    #                 elif cnt_yellow_right / (cnt_yellow_left + cnt_yellow_right + 1) > 0.1:
    #                     ava_mode = 1
    #         elif cy_white == -1 and cy_yellow > -1:
    #             ava_mode = 1
    #         elif cy_white > -1 and cy_yellow == -1:
    #             ava_mode = 2
    #         else:
    #             ava_mode = 3

    #         # Уточняем режим движения в зависимости от текущего задания
    #         if self.mode == 0:
    #             real_mode = ava_mode
    #         elif self.mode == 1:
    #             if ava_mode == 2:
    #                 real_mode = ava_mode
    #             else:
    #                 real_mode = self.mode
    #         elif self.mode == 2:
    #             if ava_mode == 1:
    #                 real_mode = ava_mode
    #             else:
    #                 real_mode = self.mode

    #         # Вычисляем угол и ошибку для управления движением
    #         angle = 0
    #         tg = 0
    #         if real_mode == 0:
    #             cv2.circle(warped_image, (cx_white, cy_white), 5, (255, 0, 0), -1)
    #             cv2.circle(warped_image, (cx_yellow, cy_yellow), 5, (0, 0, 255), -1)
    #             cv2.line(warped_image, (cx_white, cy_white), (cx_yellow, cy_yellow), (0, 255, 0), 2)

    #             cx_center = (cx_white + cx_yellow) // 2
    #             cy_center = (cy_white + cy_yellow) // 2

    #             cx_first = int((first_cx_white + first_cx_yellow) / 2)
    #             cy_first = int((first_cy_white + first_cy_yellow) / 2)
    #             cx_last = int((last_cx_white + last_cx_yellow) / 2)
    #             cy_last = int((last_cy_white + last_cy_yellow) / 2)

    #             angle = abs((cx_first - cx_last) / (cy_first - cy_last + 1))

    #             cv2.circle(warped_image, (cx_center, cy_center), 5, (0, 255, 0), -1)
    #             # cv2.imshow("Center", warped_image)
    #             # cv2.waitKey(1)

    #             error = (width // 2 - cx_center)
    #             tg = abs((cy_white - cy_yellow) / (cx_white - cx_yellow))

    #         elif real_mode == 1:
    #             cv2.circle(warped_image, (cx_yellow + 320, 240), 5, (255, 0, 0), -1)
    #             cv2.circle(warped_image, (cx_yellow, cy_yellow), 5, (0, 0, 255), -1)
    #             cv2.line(warped_image, (cx_yellow + 320, 240), (cx_yellow, cy_yellow), (0, 255, 0), 2)

    #             cx_center = cx_yellow + 160
    #             cy_center = (cy_yellow + 240) // 2

    #             cx_first = int(first_cx_yellow + 160)
    #             cy_first = int(first_cy_yellow // 2) 
    #             cx_last = int(last_cx_yellow + 160)
    #             cy_last = int((last_cy_yellow + 480) // 2)

    #             angle = abs((cx_first - cx_last) / (cy_first - cy_last + 1))

    #             # cv2.circle(warped_image, (cx_center, cy_center), 5, (0, 255, 0), -1)
    #             # cv2.imshow("Center", warped_image)
    #             cv2.waitKey(1)

    #             error = (width // 2 - cx_center)
    #             tg = abs((240 - cy_yellow) / 320)

    #         elif real_mode == 2:
    #             cv2.circle(warped_image, (cx_white, cy_white), 5, (255, 0, 0), -1)
    #             cv2.circle(warped_image, (cx_white - 320, 240), 5, (0, 0, 255), -1)
    #             cv2.line(warped_image, (cx_white - 320, 240), (cx_white, cy_white), (0, 255, 0), 2)

    #             cx_center = cx_white - 160
    #             cy_center = (240 + cy_white) // 2

    #             cx_first = int(first_cx_white - 160)
    #             cy_first = int(first_cy_white // 2) 
    #             cx_last = int(last_cx_white - 160)
    #             cy_last = int((last_cy_white + 480) // 2)

    #             angle = abs((cx_first - cx_last) / (cy_first - cy_last + 1))

    #             cv2.circle(warped_image, (cx_center, cy_center), 5, (0, 255, 0), -1)
    #             # cv2.imshow("Center", warped_image)
    #             # cv2.waitKey(1)

    #             error = (width // 2 - cx_center)
    #             tg = abs((240 - cy_white) / 320)

    #         else:
    #             error = 0

    #         # Отправляем команду на движение с учетом вычисленных параметров
    #         self.send_movement_command(error, tg, angle)
        
    #     cv2.imshow("YOLOv8 Detection", cv_image)
    #     cv2.waitKey(1)

    def send_finish_message(self, message_text):
        message = String()
        message.data = message_text
        self.finish_publisher.publish(message)
        self.get_logger().info(f'Сообщение "{message_text}" отправлено в топик /robot_finish.')

    # # def detect_objects_with_yolo(self, cv_image):
    # #     results = self.model(cv_image, verbose=False, conf=0.96)
    # #     labels = []

    # #     for result in results:
    # #         boxes = result.boxes
    # #         for box in boxes:
    # #             x1, y1, x2, y2 = box.xyxy[0]
    # #             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # #             cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # #             label = result.names[int(box.cls)]
                
    # #             confidence = box.conf.item()  # Получаем точность распознавания
    # #             if x2 > 140 and x2 < 700:
    # #                 labels.append(label)
    # #             cv2.putText(cv_image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # #             if label == "works_sign" and self.current_task > 1:
    # #                 self.lidar_3_task = True  
    # #             if label == "parking_sign" and confidence > 0.989 and self.temp_4_task and not self.finish:
    # #                 self.finish = True
    # #                 self.send_finish_message("ROSticsv8")

    # #     return cv_image, labels


    # def send_movement_command(self, error, tg, angle):
        # Получение параметров PID
        # Kp = self.get_parameter("Kp").get_parameter_value().double_value
        # Ki = self.get_parameter("Ki").get_parameter_value().double_value
        # Kd = self.get_parameter("Kd").get_parameter_value().double_value
        # desiredV = self.get_parameter("desiredV").get_parameter_value().double_value
        # # Вычисление ошибки
        # e_P = error
        # e_I = sum(self.E) + error
        # e_D = error - self.old_e


        # # Вычисление угловой скорости
        # w = Kp * e_P + Ki * e_I + Kd * e_D
        # w_max = 0.3

        # # Ограничение угловой скорости
        # if w > w_max:
        #     w = w_max
        # elif w < -w_max:
        #     w = -w_max

        # # Обновление стека интегральной ошибки
        # self.E.pop(0)
        # self.E.append(error)
        # self.old_e = error
        
        # # Динамическое управление линейной скоростью
        # # Уменьшаем линейную скорость пропорционально угловой скорости
        # angle_loss = 4
        # angular_loss = 2
        # velocity_tresh = 0.05
        # linear_velocity = max(velocity_tresh, desiredV / (np.exp(angular_loss * abs(w) / w_max) * np.exp(angle_loss * angle)))

        # # Обновление команды управления
        # twist = Twist()
        # twist.linear.x = linear_velocity
        # twist.angular.z = float(w)

        # self.publisher.publish(twist)
    
    def send_stop_command(self):
        if not self.stopped:  # Проверяем, была ли уже отправлена команда на остановку
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)

            self.stopped = True  # Устанавливаем флаг, что команда на остановку уже отправлена

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = RobotControlNode()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()