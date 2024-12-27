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
        
        # Использование весов обученной модели
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

    def usage_callback(self, msg):
        # if msg.data:
        #     self.get_logger().info('YOLO is in use')            
        # else:
        #     self.get_logger().info('YOLO is NOT in use')
        
        self.get_logger().info('YOLO is in use' if msg.data else 'YOLO is NOT in use')
        self.use_yolo = msg.data

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        if self.use_yolo:
            self.process_image(cv_image)
        else:
            cv2.imshow("Camera", cv_image)
            cv2.waitKey(1)

    def process_image(self, cv_image):
        cv_image, labels = self.detect_objects_with_yolo(cv_image)
        
        # Сохраняем обнаруженные лейблов в список
        self.last_labels.append(labels)

        # Преобразуем список лейблов в плоский список
        flat_list = [item for sublist in self.last_labels for item in sublist]

        # Сообщение с обнаруженным лейблом
        msg = String()
        msg.data = ''
        
        # Обрабатываем обнаруженные знаки
        # if "right_sign" in flat_list:
        # print(flat_list)
        # count_1 = sum([1 if i == "right_sign" else 0 for i in flat_list])
        # count_2 = sum([1 if i == "left_sign" else 0 for i in flat_list])

        # self.get_logger().info(f"{count_1}, {count_2}")

        if self.flag and sum([1 if i == "right_sign" else 0 for i in flat_list]) >= 3:
            msg.data = 'right'
            self.flag = False
            self.label_publisher.publish(msg)
        # elif "left_sign" in flat_list:
        elif self.flag and sum([1 if i == "left_sign" else 0 for i in flat_list]) >= 3:
            msg.data = 'left'
            self.flag = False
            self.label_publisher.publish(msg)
        # elif not ("right_sign" in flat_list or "left_sign" in flat_list):
        # elif sum([0 if i == "T_crossroad" else 1 for i in self.last_labels]) <= 1: 
        #     msg.data = 'cross'
        elif (not self.flag) and sum([1 if (i == "right_sign" or i == "left_sign") else 0 for i in flat_list]) == 0:
            msg.data = 'lost'
            self.flag = True
            self.label_publisher.publish(msg)
        # elif self.lidar_3_task == True and self.current_task == 3:
        #     self.current_task = 3
        elif self.work_flag and sum([1 if i == "works_sign" else 0 for i in flat_list]) >= 5:
            msg.data = 'work'
            self.label_publisher.publish(msg)
        # elif "parking_sign" in flat_list:
        #     msg.data = 'parking'
        # elif "hummer_back" in flat_list and not "works_sign" in flat_list:
        #     self.work_flag = True
        
        # self.lebeled_image_publisher.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8"))
        cv2.imshow("Camera", cv_image)
        cv2.waitKey(1)

    def detect_objects_with_yolo(self, cv_image):
        results = self.model(cv_image, verbose=False, conf=0.96)
        labels = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = result.names[int(box.cls)]
                
                confidence = box.conf.item()  # Получаем точность распознавания
                if x2 > 140 and x2 < 700:
                    labels.append(label)
                cv2.putText(cv_image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                self.get_logger().info(f"{label}, {confidence}")

                if label == "works_sign" and confidence > 0.985:
                    msg = String()
                    msg.data = 'work'
                    self.label_publisher.publish(msg)

                # if label == "works_sign" and self.current_task > 1:
                #     self.lidar_3_task = True  
                if label == "parking_sign" and confidence > 0.98:
                    self.finish = True
                    self.send_finish_message("ROSticsv8")

        return cv_image, labels
    




# rclpy.logging.get_logger("Task 1 Node").info('Message was sent. Quit working...')
# raise SystemExit


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