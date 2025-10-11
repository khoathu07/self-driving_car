#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from car_msgs.msg import CarCommand
import time
from std_msgs.msg import Bool

class LaneController:
    def __init__(self):
        # Khởi tạo node
        rospy.init_node('lane_controller', anonymous=True)
        
        # Publisher để gửi lệnh điều khiển
        self.servo_pub = rospy.Publisher('/car_control/servo', CarCommand, queue_size=10)
        self.rpm_pub = rospy.Publisher('/car_control/rpm', CarCommand, queue_size=10)
        
        # Tốc độ mặc định
        self.default_rpm = 150  # RPM mặc định cho góc lái nhỏ
        self.large_angle_rpm = 150  # RPM cho góc lái lớn
        self.angle_threshold = 30.0  # Ngưỡng để phân biệt góc lái lớn/nhỏ
        self.last_detection_time= 0
        self.detection_timeout = 2.0  # Timeout sau 2 giây
        # Đăng ký subscriber để nhận góc lái từ lane detector
        rospy.Subscriber('/lane_detector/steering_angle', CarCommand, self.lane_steering_callback)
        
        # Đăng ký subscriber để nhận góc lái từ traffic sign detector
        rospy.Subscriber('/traffic_sign/steering_angle', CarCommand, self.traffic_sign_callback)
        
        # Đăng ký subscriber để nhận diện biển stop
        rospy.Subscriber('/stop_sign/area', CarCommand, self.stop_sign_callback)
        
        # Đăng ký subscriber để nhận trạng thái lane
        rospy.Subscriber('/lane_detector/lane_status', Bool, self.lane_status_callback)
        
        # Biến lưu trữ góc lái hiện tại
        self.lane_angle = 0.0
        self.traffic_sign_angle = 0.0
        self.traffic_sign_rpm = self.default_rpm
        
        # Flag để kiểm soát ưu tiên
        self.traffic_sign_active = False
        self.traffic_sign_timeout = 2.0  # Timeout sau 2 giây
        self.last_traffic_sign_time = 0
        
        # Flag cho trạng thái dừng
        self.is_stopped = False
        
        # Góc quẹo cho biển báo
        self.turn_right_angle = 45.0
        self.turn_left_angle = -45.0
        
        # --- Parking logic ---
        self.stop_detected = False
        self.parking_detected = False  # Thêm biến này
        self.stop_area = 0
        self.area_threshold = 1200  # cập nhật ngưỡng area
        self.slow_rpm = 150
        self.lane_detected = True
        self.last_lane_time = time.time()
        self.lane_timeout = 1.0
        self.current_rpm = self.default_rpm
        # Parking state
        self.parking_mode = False
        self.parking_phase = 0  # 0: not parking, 1: forward, 2: backward, 3: backward right, 4: forward left, 5: done
        self.parking_start_time = 0
        self.parking_stop_seen_time = 0
        self.parking_last_stop_time = 0
        self.parking_waiting_exit = False
        self.delay_parking_step = 0.2  # delay giữa các lần gửi lệnh trong parking
        # --- Start journey state ---
        self.start_journey_mode = False
        self.start_journey_phase = 0  # 0: not started, 1: back, 2: right, 3: left, 4: done
        self.start_journey_start_time = 0
        
        # Tần suất gửi lệnh (Hz)
        self.rate = rospy.Rate(10)
        
        rospy.loginfo("Lane Controller initialized")

    def stop_sign_callback(self, msg):
        # msg.angle: 0 nếu có biển stop, 1 nếu không có biển, 2 nếu có biển parking
        # msg.rpm: area của biển
        if msg.angle == 2.0:
            self.parking_detected = True
            self.stop_detected = False
            self.stop_area = msg.rpm
            # Chỉ khởi tạo parking mode nếu chưa đang trong chế độ parking VÀ area đủ lớn
            if not self.parking_mode and self.stop_area >= self.area_threshold:
                self.parking_mode = True
                self.parking_phase = 1
                self.parking_start_time = time.time()
                self.parking_stop_seen_time = time.time()
                rospy.loginfo("Received parking sign, entering parking mode!")
            elif self.parking_mode:
                # Nếu đã đang trong parking mode, chỉ cập nhật thời gian thấy biển stop
                self.parking_stop_seen_time = time.time()
            else:
                # Nếu area chưa đủ lớn, chỉ cập nhật thông tin nhưng không bắt đầu parking
                rospy.loginfo("Parking sign detected but area small") 
        elif msg.angle == 0.0:
            self.parking_detected = False
            self.stop_detected = True
            self.stop_area = msg.rpm
            self.is_stopped = True
            rospy.loginfo("Received stop sign, stopping vehicle!")
        else:
            self.parking_detected = False
            self.stop_detected = False
            self.stop_area = 0
            self.is_stopped = False

    def lane_steering_callback(self, msg):
        """
        Callback function để nhận góc lái từ lane detector
        """
        self.lane_angle = msg.angle
        self.last_lane_time = time.time()
        rospy.loginfo(f"Received lane steering angle: {self.lane_angle}")

    def traffic_sign_callback(self, msg):
        """
        Callback function để nhận thông tin từ traffic sign detector
        """
        # Xác định góc quẹo dựa trên loại biển báo
        if msg.angle == 1.0:  # Biển quẹo phải
            self.traffic_sign_angle = self.turn_right_angle
            self.traffic_sign_rpm = 150
            self.is_stopped = False
            self.traffic_sign_active = True
            self.last_traffic_sign_time = time.time()
        elif msg.angle == -1.0:  # Biển quẹo trái
            self.traffic_sign_angle = self.turn_left_angle
            self.traffic_sign_rpm = 150
            self.is_stopped = False
            self.traffic_sign_active = True
            self.last_traffic_sign_time = time.time()
        elif msg.angle == 0.0:  # Không có biển báo
            self.traffic_sign_angle = 0.0
            self.traffic_sign_rpm = self.default_rpm
            self.is_stopped = False
            self.traffic_sign_active = False
            
        rospy.loginfo(f"Received traffic sign command - Angle: {self.traffic_sign_angle}, RPM: {self.traffic_sign_rpm}, Stopped: {self.is_stopped}")

    def lane_status_callback(self, msg):
        self.lane_detected = msg.data

    def get_rpm_for_angle(self, angle):
        """
        Xác định RPM dựa trên góc lái
        """
        if abs(angle) > self.angle_threshold:
            return self.large_angle_rpm
        return self.default_rpm

    def send_control_commands(self):
        """
        Gửi lệnh điều khiển đến ESP32
        """
        while not rospy.is_shutdown():
            try:
                now = time.time()
                # Kiểm tra mất lane
                if now - self.last_lane_time > self.lane_timeout:
                    self.lane_detected = False
                # Ưu tiên dừng xe khi phát hiện biển stop
                if self.is_stopped:
                    cmd = CarCommand()
                    cmd.angle = 0.0
                    cmd.rpm = 0
                    rospy.loginfo("Vehicle stopped due to stop sign!")
                    self.servo_pub.publish(cmd)
                    self.rpm_pub.publish(cmd)
                    self.rate.sleep()
                    continue
                # PARKING LOGIC
                if (self.parking_detected and self.stop_area >= self.area_threshold and not self.parking_mode):
                    self.parking_mode = True
                    self.parking_phase = 1  
                    self.parking_start_time = now
                    self.parking_stop_seen_time = now
                    rospy.loginfo("Entering parking mode: phase 1 (forward)")
                
                if self.parking_mode:
                    cmd = CarCommand()
                   
                    if self.parking_phase == 1:
                        cmd.angle = 0.0
                        cmd.rpm = 150
                        if (now - self.parking_start_time) >= 11.0:
                            self.parking_phase = 2
                            self.parking_start_time = now
                            rospy.loginfo("Parking phase 2: backward with -45 angle")
                        self.servo_pub.publish(cmd)
                        self.rpm_pub.publish(cmd)
                        time.sleep(self.delay_parking_step)
                        self.rate.sleep()
                        continue
                    
                    if self.parking_phase == 2:
                        cmd.angle = -45.0
                        cmd.rpm = -150
                        if (now - self.parking_start_time) >= 5.0:
                            self.parking_phase = 3
                            self.parking_start_time = now
                            rospy.loginfo("Parking phase 3: backward right 45 deg 3.5s")
                        self.servo_pub.publish(cmd)
                        self.rpm_pub.publish(cmd)
                        time.sleep(self.delay_parking_step)
                        self.rate.sleep()
                        continue
                    
                    if self.parking_phase == 3:
                        cmd.angle = 45.0
                        cmd.rpm = -150
                        if (now - self.parking_start_time) >= 3.5:
                            self.parking_phase = 4
                            self.parking_start_time = now
                            rospy.loginfo("Parking phase 4: forward left -30 deg 1s")
                        self.servo_pub.publish(cmd)
                        self.rpm_pub.publish(cmd)
                        time.sleep(self.delay_parking_step)
                        self.rate.sleep()
                        continue
                    
                    if self.parking_phase == 4:
                        cmd.angle = -30.0
                        cmd.rpm = 150
                        if (now - self.parking_start_time) >= 1.0:
                            self.parking_phase = 5
                            self.parking_start_time = now
                            rospy.loginfo("Parking phase 5: stopped, waiting for stop sign lost 10s to exit")
                        self.servo_pub.publish(cmd)
                        self.rpm_pub.publish(cmd)
                        time.sleep(self.delay_parking_step)
                        self.rate.sleep()
                        continue
                    
                    if self.parking_phase == 5:
                        cmd.angle = 0.0
                        cmd.rpm = 0
                        
                        if not (self.parking_detected and self.stop_area >= self.area_threshold):
                            if (now - self.parking_stop_seen_time) > 5.0:
                                self.parking_mode = False
                                self.parking_phase = 0
                                rospy.loginfo("Exiting parking mode: stop sign lost for 10s after parking complete. Start journey!")
                                self.start_journey_mode = True
                                self.start_journey_phase = 1
                                self.start_journey_start_time = now
                                continue
                        else:
                            self.parking_stop_seen_time = now  
                        self.servo_pub.publish(cmd)
                        self.rpm_pub.publish(cmd)
                        time.sleep(self.delay_parking_step)
                        self.rate.sleep()
                        continue
                #  END PARKING LOGIC 
                #  START JOURNEY LOGIC 
                if self.start_journey_mode:
                    cmd = CarCommand()
                   
                    if self.start_journey_phase == 1:
                        cmd.angle = 0.0
                        cmd.rpm = -150
                        if (now - self.start_journey_start_time) >= 1.0:
                            self.start_journey_phase = 2
                            self.start_journey_start_time = now
                            rospy.loginfo("Start journey phase 2: right 2s")
                        self.servo_pub.publish(cmd)
                        self.rpm_pub.publish(cmd)
                        time.sleep(self.delay_parking_step)
                        self.rate.sleep()
                        continue
                    
                    if self.start_journey_phase == 2:
                        cmd.angle = 45.0
                        cmd.rpm = 150
                        if (now - self.start_journey_start_time) >= 3.0:
                            self.start_journey_phase = 3
                            self.start_journey_start_time = now
                            rospy.loginfo("Start journey phase 3: left 4s")
                        self.servo_pub.publish(cmd)
                        self.rpm_pub.publish(cmd)
                        time.sleep(self.delay_parking_step)
                        self.rate.sleep()
                        continue
                    
                    if self.start_journey_phase == 3:
                        cmd.angle = -45.0
                        cmd.rpm = 150
                        if (now - self.start_journey_start_time) >= 4.5:
                            self.start_journey_phase = 4
                            self.start_journey_start_time = now
                            rospy.loginfo("Start journey phase 4: done, switch to lane following")
                        self.servo_pub.publish(cmd)
                        self.rpm_pub.publish(cmd)
                        time.sleep(self.delay_parking_step)
                        self.rate.sleep()
                        continue
                    
                    if self.start_journey_phase == 4:
                        cmd.angle = 0.0
                        cmd.rpm = -150
                        if (now - self.start_journey_start_time) >= 3.0:
                            cmd.angle = 45.0
                            cmd.rpm = 150
                        if (now - self.start_journey_start_time) >= 9.0:
                            self.start_journey_phase = 5
                            self.start_journey_start_time = now
                            rospy.loginfo("Start journey phase 5: done, switch to lane following")
                        self.servo_pub.publish(cmd)
                        self.rpm_pub.publish(cmd)
                        time.sleep(self.delay_parking_step)
                        self.rate.sleep()
                        continue
                
                    if self.start_journey_phase == 5:
                        self.start_journey_mode = False
                        self.start_journey_phase = 0
                        rospy.loginfo("Start journey complete. Resume lane following.")
                    
                # --- END START JOURNEY LOGIC ---
                
                cmd = CarCommand()
                if self.traffic_sign_active and not self.is_stopped and (time.time() - self.last_traffic_sign_time) > self.detection_timeout:
                    self.traffic_sign_active = False
                    rospy.loginfo("timeout, switching to lane detection")
                if self.traffic_sign_active:
                    cmd.angle = self.traffic_sign_angle
                    cmd.rpm = self.traffic_sign_rpm
                    rospy.loginfo(f"Using traffic sign control - Angle: {cmd.angle}, RPM: {cmd.rpm}")
                else:
                    if not self.lane_detected:
                        cmd.angle = self.lane_angle
                        cmd.rpm = self.get_rpm_for_angle(self.lane_angle)
                        rospy.loginfo(f"Lane lost in normal mode, using last angle: {cmd.angle}")
                    else:
                        cmd.angle = self.lane_angle
                        cmd.rpm = self.get_rpm_for_angle(self.lane_angle)
                        rospy.loginfo(f"Using lane detection control - Angle: {cmd.angle}, RPM: {cmd.rpm}")
                self.servo_pub.publish(cmd)
                self.rpm_pub.publish(cmd)
                self.rate.sleep()
            except Exception as e:
                rospy.logerr(f"Error sending commands: {e}")
                continue

if __name__ == '__main__':
    try:
        controller = LaneController()
        controller.send_control_commands()
    except rospy.ROSInterruptException:
        pass