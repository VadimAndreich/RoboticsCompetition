import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
import cv2
import os
import time
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from collections import deque

class YOLONode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        # Подписки на топики
        self.usage_subscriber = self.create_subscription(Bool, '/yolo/use_yolo', self.usage_callback, 10)   # Флаг использования yolo
        self.image_subscriber = self.create_subscription(Image, '/color/image', self.image_callback, 10)    # Изображения с камеры
        
        # Публикация в топик
        self.label_publisher = self.create_publisher(String, '/yolo/label', 10)         # Обнаруженные объекты
        self.lebeled_image_publisher = self.create_publisher(Image, '/yolo/image', 10)  # Изображение с bounding box-ами
        self.finish_publisher = self.create_publisher(String, '/robot_finish', 10)      # Сообщение об остановке возле знака парковки
        
        # Использование весов обученной модели yolov8
        package_name = 'autorace_core_ROSticsv8'
        package_share_dir = get_package_share_directory(package_name)
        weights_path = os.path.join(package_share_dir, 'weights', 'ros.pt')
        
        self.bridge = CvBridge()

        self.model = YOLO(weights_path)
        self.use_yolo = False
        self.last_labels = deque(maxlen = 7)

        self.flag = True
        self.work_flag = True
        self.slowed_flag = True
        self.works = 0
        self.time = 0
        self.seconds = 0

    def usage_callback(self, msg):
        self.get_logger().info('YOLO is in use' if msg.data else 'YOLO is NOT in use')
        self.use_yolo = msg.data
        
        if self.works == 0:
            self.seconds = 12
            self.time = time.time()
        
        self.works += 1

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.seconds != 0:
            if time.time() - self.time > self.seconds:
                self.seconds = 0
            cv2.imshow("Camera", cv_image)
            cv2.waitKey(1)
        
        # Обрабатываем изображение с помощью yolo
        elif self.use_yolo:
            self.process_image(cv_image)
        else:
            cv2.imshow("Camera", cv_image)
            cv2.waitKey(1)

    def disable_yolo(self, seconds):
        if time.time() - self.time > seconds:
            self.use_yolo = True

    def process_image(self, cv_image):
        cv_image, labels = self.detect_objects_with_yolo(cv_image)
        
        # Получаем список последних обнаруженных лейблов
        self.last_labels.append(labels)
        labels_list = [item for sublist in self.last_labels for item in sublist]

        msg = String()
        msg.data = ''
        
        # count_1 = sum([1 if i == "right_sign" else 0 for i in flat_list])
        # count_2 = sum([1 if i == "left_sign" else 0 for i in flat_list])

        # self.get_logger().info(f"{count_1}, {count_2}")

        if self.flag and sum([1 if i == "right_sign" else 0 for i in labels_list]) >= 3:
            # Знак вправо
            msg.data = 'right'
            self.flag = False
            self.label_publisher.publish(msg)
        elif self.flag and sum([1 if i == "left_sign" else 0 for i in labels_list]) >= 3:
            # Знак влево
            msg.data = 'left'
            self.flag = False
            self.label_publisher.publish(msg)
        elif (not self.flag) and sum([1 if (i == "right_sign" or i == "left_sign") else 0 for i in labels_list]) == 0:
            # Знак потерян после обнаружения
            msg.data = 'lost'
            self.flag = True
            self.label_publisher.publish(msg)

            self.seconds = 30
            self.time = time.time()

        elif self.work_flag and sum([1 if i == "works_sign" else 0 for i in labels_list]) >= 3:
            # Знак дорожных работ
            msg.data = 'work'
            self.label_publisher.publish(msg)

        cv2.imshow("Camera", cv_image)
        cv2.waitKey(1)


    def detect_objects_with_yolo(self, cv_image):
        # Получаем результаты работы модели с confidence > 0.96
        results = self.model(cv_image, verbose=False, conf=0.96)
        labels = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Координаты bbox
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Вывод bbox и лейбла
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = result.names[int(box.cls)]
                
                # Вывод confidence
                confidence = box.conf.item()
                if x2 > 140 and x2 < 700:
                    labels.append(label)    
                
                cv2.putText(cv_image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # self.get_logger().info(f"{label}, {confidence}")

                # Обрабатываем найденный знак дорожных работ
                if label == "works_sign" and confidence > 0.985:
                    msg = String()
                    msg.data = 'work'
                    self.label_publisher.publish(msg)

                # Обрабатываем найденный знак парковки
                if label == "parking_sign" and confidence > 0.98 and self.works > 1:
                    self.finish = True

                    # Заканчиваем работу робота
                    msg = String()
                    msg.data = "ROSticsv8"
                    self.finish_publisher.publish(msg)

                    self.use_yolo = False
                    # rclpy.logging.get_logger("YOLO Node").info('Message was sent. Quit working...')
                    # raise SystemExit

        return cv_image, labels


def main(args=None):
    rclpy.init(args=args)
    yolo = YOLONode()
    
    rclpy.logging.get_logger("YOLO Node").info('Working...')

    try:
        rclpy.spin(yolo)
    except SystemExit:
        rclpy.logging.get_logger("YOLO Node").info('Done!')
    finally:
        yolo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()