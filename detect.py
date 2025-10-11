#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import time
import torch
import numpy as np
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from car_msgs.msg import CarCommand

# Thêm đường dẫn tới thư mục yolov5 (clone repo vào ~/yolov5)
sys.path.append('/home/jetson/yolov5')  
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class YoloPtNode:
    def __init__(self):
        rospy.init_node('yolo_pt_node', anonymous=True)
        self.bridge = CvBridge()
        self.last_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.fps_update_interval = 0.5  # Cập nhật FPS mỗi 0.5 giây
        self.fps_last_update = time.time()

        # Chuẩn bị device
        device = select_device('0' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {device}")

        # Load model bằng attempt_load
        model_path = rospy.get_param('model_path', r'/home/jetson/Desktop/best_750.pt')
        self.model = attempt_load(model_path, device=device)
        self.model.to(device).eval()

        # Lấy tên class từ model
        self.class_names = ['left', 'right', 'stop','parking']

        # Subscribe ảnh
        self.sub = rospy.Subscriber(
            'video_topic', Image, self.callback,
            queue_size=1, buff_size=2**24)
            
        # Publisher cho traffic sign detection
        self.traffic_sign_pub = rospy.Publisher('/traffic_sign/steering_angle', CarCommand, queue_size=10)
        
        # Publisher cho stop sign area
        self.stop_sign_pub = rospy.Publisher('/stop_sign/area', CarCommand, queue_size=10)
        
        # Ngưỡng diện tích cho traffic sign
        self.area_threshold = 3000  # Có thể điều chỉnh giá trị này
        
        # Flag để kiểm soát ưu tiên
        self.traffic_sign_active = False
        self.last_detection_time = 0
        self.detection_timeout = 2.0  # Timeout sau 2 giây
        
        # Thêm biến cho voting và pending vote (chỉ cho left/right)
        self.voting_active = False
        self.voting_labels = []
        self.voting_frame_count = 0
        self.voting_area_threshold = 1000
        self.voting_max_frames = 11
        self.pending_vote = None
        self.pending_vote_sent = False

    def update_fps(self):
        """
        Cập nhật FPS mỗi khoảng thời gian nhất định
        """
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.fps_last_update >= self.fps_update_interval:
            self.fps = self.frame_count / (current_time - self.fps_last_update)
            self.frame_count = 0
            self.fps_last_update = current_time

    def callback(self, msg):
        # Cập nhật FPS
        self.update_fps()

        # Chuyển đổi hình ảnh từ ROS sang OpenCV
        try:
            img0 = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Chuyển đổi BGR sang RGB
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

        # Thay đổi kích thước hình ảnh với letterbox
        img_resized = letterbox(img, new_shape=640)[0]

        # Chuyển đổi hình ảnh sang định dạng phù hợp cho mô hình
        img_trans = img_resized.transpose((2, 0, 1))[None]  # BCHW
        img_trans = np.ascontiguousarray(img_trans) / 255.0
        img_tensor = torch.from_numpy(img_trans).float()

        # Lấy thiết bị của mô hình
        device = next(self.model.parameters()).device
        img_tensor = img_tensor.to(device)

        # Thực hiện suy luận
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
            pred = non_max_suppression(pred, 0.25, 0.45)

        # Biến để kiểm tra có phát hiện biển báo stop không
        detected_label = None
        detected_area = 0
        detected_parking = False

        # Xử lý kết quả và vẽ lên hình ảnh
        for det in pred:
            if det is not None and len(det):
                # Chuyển đổi tọa độ hộp giới hạn về kích thước gốc của hình ảnh
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det.cpu().numpy():
                    x1, y1, x2, y2 = map(int, xyxy)
                    area = (x2 - x1) * (y2 - y1)
                    label = self.class_names[int(cls)]
                    if label == 'parking':
                        detected_label = 'parking'
                        detected_area = area
                        detected_parking = True
                    elif label == 'stop' and area > 2000:
                        detected_label = 'stop'
                        detected_area = area
                    # Vẽ bounding box và nhãn
                    cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img0, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(img0, f"Area: {area}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Xử lý parking sign - gửi thông tin qua topic /stop_sign/area
        if detected_label == "parking":
            cmd = CarCommand()
            cmd.angle = 2.0  # 2 = có biển parking
            cmd.rpm = detected_area
            self.stop_sign_pub.publish(cmd)
            rospy.loginfo(f"Parking sign detected - area: {detected_area}")
        # Xử lý stop sign - gửi thông tin qua topic /stop_sign/area
        elif detected_label == "stop" and detected_area > 2000:
            cmd = CarCommand()
            cmd.angle = 0.0  # 0 = có biển stop
            cmd.rpm = detected_area  # Gửi area của biển stop
            self.stop_sign_pub.publish(cmd)
            rospy.loginfo(f"Stop sign detected - area: {detected_area}")
        elif detected_label not in ["stop", "parking"] or detected_area <= 2000:
            cmd = CarCommand()
            cmd.angle = 1.0  # 1 = không có biển stop/parking
            cmd.rpm = 0
            self.stop_sign_pub.publish(cmd)
            rospy.loginfo("Stop/Parking sign no longer detected")

        # Voting logic chỉ cho left/right signs
        if not self.voting_active and detected_label in ["right", "left"] and detected_area > self.voting_area_threshold:
            self.voting_active = True
            self.voting_labels = []
            self.voting_frame_count = 0
            rospy.loginfo("Start voting for direction!")

        if self.voting_active:
            if detected_label in ["right", "left"] and detected_area > self.voting_area_threshold:
                self.voting_labels.append(detected_label)
            else:
                self.voting_labels.append(None)
            self.voting_frame_count += 1
            if self.voting_frame_count >= self.voting_max_frames:
                right_count = self.voting_labels.count("right")
                left_count = self.voting_labels.count("left")
                if right_count > left_count:
                    self.pending_vote = "right"
                    rospy.loginfo(f"Voting result: right ({right_count} vs {left_count}) - pending execution")
                elif left_count > right_count:
                    self.pending_vote = "left"
                    rospy.loginfo(f"Voting result: left ({left_count} vs {right_count}) - pending execution")
                else:
                    self.pending_vote = None
                    rospy.loginfo(f"Voting result: tie ({right_count} vs {left_count}), no action")
                self.voting_active = False
                self.voting_labels = []
                self.voting_frame_count = 0
                self.pending_vote_sent = False

        # Chỉ gửi lệnh khi đã có kết quả vote và area > 3000
        if self.pending_vote and detected_area > self.area_threshold and not self.pending_vote_sent:
            cmd = CarCommand()
            if self.pending_vote == "right":
                cmd.angle = 1.0
                cmd.rpm = 150
            elif self.pending_vote == "left":
                cmd.angle = -1.0
                cmd.rpm = 150
            self.traffic_sign_pub.publish(cmd)
            rospy.loginfo(f"Pending vote executed: {self.pending_vote} (area={detected_area})")
            self.pending_vote_sent = True
            self.pending_vote = None
        
        # Kiểm tra timeout cho traffic sign detection (chỉ áp dụng cho biển báo left/right)
        if self.traffic_sign_active and (time.time() - self.last_detection_time) > self.detection_timeout:
            self.traffic_sign_active = False
            rospy.loginfo("Traffic sign detection timeout")
        
        # Vẽ FPS lên hình ảnh
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(img0, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow("YOLOv5 Detection", img0)
        cv2.waitKey(1)

    def spin(self):
        rospy.spin()
        cv2.destroyAllWindows()

def letterbox(img, new_shape=640, color=(114,114,114)):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = img.shape[:2]  # h, w
    r = new_shape / max(shape)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2; dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, dw, dh

if __name__ == '__main__':
    try:
        node = YoloPtNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
