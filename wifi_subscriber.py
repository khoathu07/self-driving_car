#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import threading
from car_msgs.msg import CarCommand
from std_msgs.msg import Bool

subscriberNodeName = 'lane_detector_subscriber'
topicName = 'video_topic'

bridge = CvBridge()
previous_time = time.time()

wT = 640
hT = 480

# Cache cho filters và transform matrix
gaussian_filter = None
canny_detector = None

def init_cuda_filters(gpu_gray):
    global gaussian_filter, canny_detector
    if gaussian_filter is None:
        gaussian_filter = cv2.cuda.createGaussianFilter(
            gpu_gray.type(), gpu_gray.type(), (5,5), 0)
    if canny_detector is None:
        canny_detector = cv2.cuda.createCannyEdgeDetector(150, 255)

class LaneDetector:
    def __init__(self):
        self.Wb = 380
        self.prev_lane_fit = None
        self.last_valid_intersection = None
        # Cache cho ma trận transform
        self._M_cpu = None
        self.lane_side = None  # 'left' hoặc 'right'
        self.leftx_base = None
        self.rightx_base = None
        self.history = []  # Lưu lại (center_x, intersection_x, timestamp)
    
    def warpImg_cuda(self, img_cpu, points, w, h):
        """
        WarpPerspective sử dụng CUDA
        """
        # Tính và cache ma trận M (numpy)
        if self._M_cpu is None:
            pts1 = np.float32(points)
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            self._M_cpu = cv2.getPerspectiveTransform(pts1, pts2).astype(np.float32)
        
        # Upload ảnh CPU lên GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img_cpu)
        
        # WarpPerspective trên GPU
        gpu_warped = cv2.cuda.warpPerspective(
            gpu_img,
            self._M_cpu,
            (w, h),
            flags=cv2.INTER_LINEAR
        )
        return gpu_warped

    def detect_yellow_lanes_cuda(self, gpu_img):
        """
        Xử lý tìm làn đường màu vàng sử dụng CUDA
        """
        # Download ảnh để xử lý HSV (vì không có hàm cvtColor HSV trên CUDA)
        img_cpu = gpu_img.download()
        hsv = cv2.cvtColor(img_cpu, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, np.array([19, 28, 170]), np.array([35, 255, 255]))
        
        # Upload mask lên GPU
        gpu_mask = cv2.cuda_GpuMat()
        gpu_mask.upload(yellow_mask)
        
        # Chuyển sang ảnh binary
        gpu_threshold = cv2.cuda.threshold(gpu_mask, 50, 255, cv2.THRESH_BINARY)[1]
        
        return gpu_threshold

    def process_image_cuda(self, gpu_img):
        """
        Xử lý ảnh sử dụng CUDA:
        - convert to gray
        - gaussian blur
        - canny
        Trả về: GPU Mat binary
        """
        global gaussian_filter, canny_detector
        
        # Chuyển sang ảnh xám
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        
        # Khởi tạo filters nếu cần
        init_cuda_filters(gpu_gray)
        
        # Áp dụng gaussian blur
        gpu_blur = gaussian_filter.apply(gpu_gray)
        
        # Canny edge detection
        gpu_edges = canny_detector.detect(gpu_blur)
        
        return gpu_edges

    def sliding_window_search(self, binary_img, nwindows=9, margin=100, minpix=50):
        # Convert từ GPU Mat sang CPU nếu cần
        if isinstance(binary_img, cv2.cuda_GpuMat):
            binary_img = binary_img.download()
            
        histogram = np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)
        out_img = np.dstack((binary_img, binary_img, binary_img)) * 255
        midpoint = histogram.shape[0]//2
        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        if histogram[self.leftx_base] > histogram[self.rightx_base]:
            lane_base = self.leftx_base
        else:
            lane_base = self.rightx_base
        
        window_height = np.int_(binary_img.shape[0]/nwindows)
        nonzero = binary_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        lane_current = lane_base
        lane_inds = []
        
        for window in range(nwindows):
            win_y_low = binary_img.shape[0] - (window+1)*window_height
            win_y_high = binary_img.shape[0] - window*window_height
            win_x_low = lane_current - margin
            win_x_high = lane_current + margin
            
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                        (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            
            lane_inds.append(good_inds)
            
            if len(good_inds) > minpix:
                lane_current = np.int_(np.mean(nonzerox[good_inds]))
        
        lane_inds = np.concatenate(lane_inds) if lane_inds else np.array([])
        
        lanex = nonzerox[lane_inds] if len(lane_inds) > 0 else np.array([])
        laney = nonzeroy[lane_inds] if len(lane_inds) > 0 else np.array([])
        
        if len(lanex) == 0 or len(laney) == 0:
            lane_fit = np.array([0, 0, 0])
            ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0])
            lane_fitx = np.zeros_like(ploty)
            pts = np.array([np.transpose(np.vstack([lane_fitx, ploty]))])
            pts = pts.astype(np.int32)
            key_points = [(0, binary_img.shape[0]-1), (0, 3*binary_img.shape[0]//4), 
                         (0, binary_img.shape[0]//2), (0, binary_img.shape[0]//4), (0, 0)]
            return out_img, pts, lane_fit, key_points
            
        lane_fit = np.polyfit(laney, lanex, 2)
        
        ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0])
        try:
            lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
        except:
            lane_fitx = 1*ploty**2 + 1*ploty
            
        pts = np.array([np.transpose(np.vstack([lane_fitx, ploty]))])
        pts = pts.astype(np.int32)
        
        h = binary_img.shape[0]
        
        far_y = 0
        middle_3_y = h // 4
        middle_2_y = h // 2
        middle_1_y = 3 * h // 4
        near_y = h - 1

        near_x = int(lane_fit[0] * near_y**2 + lane_fit[1] * near_y + lane_fit[2])
        middle_1_x = int(lane_fit[0] * middle_1_y**2 + lane_fit[1] * middle_1_y + lane_fit[2])
        middle_2_x = int(lane_fit[0] * middle_2_y**2 + lane_fit[1] * middle_2_y + lane_fit[2])
        middle_3_x = int(lane_fit[0] * middle_3_y**2 + lane_fit[1] * middle_3_y + lane_fit[2])
        far_x = int(lane_fit[0] * far_y**2 + lane_fit[1] * far_y + lane_fit[2])
        
        near_point = (near_x, near_y)
        middle_1_point = (middle_1_x, middle_1_y)
        middle_2_point = (middle_2_x, middle_2_y)
        middle_3_point = (middle_3_x, middle_3_y)
        far_point = (far_x, far_y)
        
        key_points = [near_point, middle_1_point, middle_2_point, middle_3_point, far_point]
        
        return out_img, pts, lane_fit, key_points

    def move_line(self, key_points):
        near_point = key_points[0]
        mid_1_point = key_points[1]
        mid_2_point = key_points[2]
        mid_3_point = key_points[3]
        far_point = key_points[4]

        if self.lane_side == 'left':
            delta_x = 265
        elif self.lane_side == 'right':
            delta_x = -265
        else:
            delta_x = 0
    
        near_point_shifted = (int(near_point[0] + delta_x), int(near_point[1]))
        mid_1_point_shifted = (int(mid_1_point[0] + delta_x), int(mid_1_point[1]))
        mid_2_point_shifted = (int(mid_2_point[0] + delta_x), int(mid_2_point[1]))
        mid_3_point_shifted = (int(mid_3_point[0] + delta_x), int(mid_3_point[1]))
        far_point_shifted = (int(far_point[0] + delta_x), int(far_point[1]))
        
        return near_point_shifted, mid_1_point_shifted, mid_2_point_shifted, mid_3_point_shifted, far_point_shifted

    def draw_circle(self, h, w):
        center = (int(w // 2), int(h))
        axes = int(w//2)
        return center, axes

    def calculate_intersection_points(self, line_points, circle_center, circle_radius):
        intersections = []
        
        for i in range(len(line_points) - 1):
            point1 = line_points[i]
            point2 = line_points[i + 1]
            
            segment_intersections = self.line_segment_circle_intersection(point1, point2, circle_center, circle_radius)
            intersections.extend(segment_intersections)
        
        return intersections

    def line_segment_circle_intersection(self, point1, point2, circle_center, circle_radius):
        point1 = np.array(point1, dtype=float)
        point2 = np.array(point2, dtype=float)
        circle_center = np.array(circle_center, dtype=float)
        
        direction = point2 - point1
        
        segment_length = np.linalg.norm(direction)
        if segment_length < 1e-10:
            return []
        
        direction = direction / segment_length
        
        start_to_center = circle_center - point1
        
        projection = np.dot(start_to_center, direction)
        
        dist = np.linalg.norm(start_to_center - projection * direction)
        
        if dist > circle_radius:
            return []
        
        offset = np.sqrt(circle_radius**2 - dist**2)
        
        t1 = projection - offset
        t2 = projection + offset
        
        intersections = []
        
        if 0 <= t1 <= segment_length:
            intersection1 = point1 + t1 * direction
            intersections.append(tuple(intersection1))
        
        if 0 <= t2 <= segment_length and abs(t1 - t2) > 1e-10:
            intersection2 = point1 + t2 * direction
            intersections.append(tuple(intersection2))
        
        return intersections

    def get_intersection_with_lane(self, center, axes, lane_points_shifted):
        circle_center = center
        circle_radius = axes
        
        intersections = self.calculate_intersection_points(lane_points_shifted, circle_center, circle_radius)
        return intersections

    def Pure_Pursuit(self, goal_point, current_point, L, WB=380):
        gx = goal_point[1]
        gy = goal_point[0]
        cx = current_point[1]
        cy = current_point[0]

        ptocm = 26

        dy = gy - cy 
        
        dy_cm = dy / ptocm
        L_cm = L / ptocm
        
        alpha = 2 * dy_cm / L_cm**2

        delta = alpha * WB

        if delta > 45:
            delta = 45
        elif delta < -45:
            delta = -45

        return float(delta)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        
        # Fixed values from original code
        intialTrackbarVals_right = [57, 272, 0, 404] #right
        intialTrackbarVals_left = [65, 369, 0, 480] #left
        
        # Use left trackbar values by default
        points = np.float32([(intialTrackbarVals_left[0], intialTrackbarVals_left[1]),
                        (wT - intialTrackbarVals_left[0], intialTrackbarVals_left[1]),
                        (intialTrackbarVals_left[2], intialTrackbarVals_left[3]),
                        (wT - intialTrackbarVals_left[2], intialTrackbarVals_left[3])])
        
        # Warp the image using CUDA
        gpu_warp = self.warpImg_cuda(frame, points, w, h)
        
        # Detect yellow lanes on GPU
        gpu_binary = self.detect_yellow_lanes_cuda(gpu_warp)
        
        # Download để sử dụng cho sliding window search
        binary_lane = gpu_binary.download()
        
        # Apply sliding window search
        lane_tracking, lane_points, lane_fit, key_points = self.sliding_window_search(binary_lane)
        
        # Sau khi đã có leftx_base và rightx_base, cập nhật lại trackbar nếu cần
        if self.leftx_base is not None and self.rightx_base is not None:
            # Lấy lại histogram từ sliding_window_search
            histogram = np.sum(binary_lane[binary_lane.shape[0]//2:, :], axis=0)
            if histogram[self.leftx_base] > histogram[self.rightx_base]:
                if self.lane_side != 'left':
                    self.lane_side = 'left'
                    rospy.loginfo("left")
            else:
                if self.lane_side != 'right':
                    self.lane_side = 'right'
                    rospy.loginfo("right")
        
        # Smooth lane detection
        if self.prev_lane_fit is not None:
            lane_fit = (lane_fit + self.prev_lane_fit) / 2
        
        self.prev_lane_fit = lane_fit
        
        # Tải ảnh từ GPU để vẽ overlay
        imgWarp_color = gpu_warp.download()
        
        # Create overlay for visualization
        lane_overlay = imgWarp_color.copy()
        lane_overlay_h, lane_overlay_w, _ = lane_overlay.shape
        
        # Extract key points
        near_point, middle_1_point, middle_2_point, middle_3_point, far_point = key_points
        
        # Draw key points and lines
        cv2.circle(lane_overlay, near_point, 10, (0, 35, 245), cv2.FILLED)
        cv2.circle(lane_overlay, middle_1_point, 10, (235, 51, 0), cv2.FILLED)
        cv2.circle(lane_overlay, middle_2_point, 10, (117, 22, 63), cv2.FILLED)
        cv2.circle(lane_overlay, middle_3_point, 10, (255, 253, 85), cv2.FILLED)
        cv2.circle(lane_overlay, far_point, 10, (117, 250, 141), cv2.FILLED)
        
        cv2.line(lane_overlay, near_point, middle_1_point, (0, 0, 0), 4)
        cv2.line(lane_overlay, middle_1_point, middle_2_point, (255, 255, 255), 4)
        cv2.line(lane_overlay, middle_2_point, middle_3_point, (0, 0, 0), 4)
        cv2.line(lane_overlay, middle_3_point, far_point, (255, 255, 255), 4)
        
        # Calculate shifted lane
        near_point_shifted, mid_1_point_shifted, mid_2_point_shifted, mid_3_point_shifted, far_point_shifted = self.move_line(key_points)
        
        # Draw shifted lane
        cv2.circle(lane_overlay, near_point_shifted, 10, (0, 35, 245), cv2.FILLED)
        cv2.circle(lane_overlay, mid_1_point_shifted, 10, (235, 51, 0), cv2.FILLED)
        cv2.circle(lane_overlay, mid_2_point_shifted, 10, (117, 22, 63), cv2.FILLED)
        cv2.circle(lane_overlay, mid_3_point_shifted, 10, (255, 253, 85), cv2.FILLED)
        cv2.circle(lane_overlay, far_point_shifted, 10, (117, 250, 141), cv2.FILLED)
        
        cv2.line(lane_overlay, near_point_shifted, mid_1_point_shifted, (0, 0, 255), 2)
        cv2.line(lane_overlay, mid_1_point_shifted, mid_2_point_shifted, (0, 0, 255), 2)
        cv2.line(lane_overlay, mid_2_point_shifted, mid_3_point_shifted, (0, 0, 255), 2)
        cv2.line(lane_overlay, mid_3_point_shifted, far_point_shifted, (0, 0, 255), 2)
        
        # Draw circular path
        center, axes = self.draw_circle(lane_overlay_h, lane_overlay_w)
        
        cv2.ellipse(
            lane_overlay,
            center,
            (axes, axes),
            0,
            180,
            360,
            (0, 225, 255),
            3
        )
        
        top_image = (int(lane_overlay_w // 2), 0)
        cv2.line(lane_overlay, center, top_image, (192, 192, 192), 2)
        
        lane_points_shifted = [
            near_point_shifted,
            mid_1_point_shifted,
            mid_2_point_shifted,
            mid_3_point_shifted,
            far_point_shifted
        ]
        
        # Calculate intersections
        intersections = self.get_intersection_with_lane(center, axes, lane_points_shifted)
        
        # Check if lane is valid (all key points should not be at x=0)
        lane_valid = all(point[0] != 0 for point in key_points)
        
        # Handle case with no intersections or invalid lane
        if not intersections or not lane_valid:
            if self.last_valid_intersection is not None and lane_valid:
                inter_x, inter_y = self.last_valid_intersection
                # Calculate steering angle with last valid intersection
                goal_point = (inter_x, inter_y)
                steering_angle = self.Pure_Pursuit(goal_point, center, axes)
            else:
                # No valid lane detected
                inter_x, inter_y = center[0], center[1] - 100
                steering_angle = None  # Signal no valid lane
        else:
            intersection_points = intersections[0]
            inter_x = int(intersection_points[0])
            inter_y = int(intersection_points[1])
            self.last_valid_intersection = (inter_x, inter_y)
            # Calculate steering angle
            goal_point = (inter_x, inter_y)
            steering_angle = self.Pure_Pursuit(goal_point, center, axes)
        
        cv2.circle(lane_overlay, (inter_x, inter_y), 6, (0, 0, 0), cv2.FILLED)
        cv2.line(lane_overlay, center, (inter_x, inter_y), (234, 54, 128), 2)
        cv2.circle(lane_overlay, center, 10, (0, 0, 255), cv2.FILLED)
        
        if steering_angle is not None:
            cv2.putText(lane_overlay, f"Steering Angle: {steering_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(lane_overlay, "No Lane Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Return processed images and steering angle
        return frame, lane_overlay, steering_angle, binary_lane, lane_valid

# Initialize the lane detector
lane_detector = LaneDetector()

# Thêm vào phần khởi tạo (sau khi khởi tạo node)
steering_pub = rospy.Publisher('/lane_detector/steering_angle', CarCommand, queue_size=10)
lane_status_pub = rospy.Publisher('/lane_detector/lane_status', Bool, queue_size=10)

def callbackFunction(msg):
    global previous_time
    rospy.loginfo("Received a video frame")
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Process frame using the lane detector with CUDA
        original, lane_overlay, steering_angle, binary, lane_valid = lane_detector.process_frame(frame)
        
        # Publish steering angle only if lane is valid
        if steering_angle is not None:
            cmd = CarCommand()
            cmd.angle = steering_angle
            steering_pub.publish(cmd)
        # Publish lane status
        lane_status_pub.publish(lane_valid)
        
        # Calculate FPS
        now = time.time()
        fps = 1.0 / (now - previous_time)
        previous_time = now
        
        # Add FPS to display
        cv2.putText(lane_overlay, f"FPS: {fps:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display results
        cv2.imshow("Original", original)
        cv2.imshow("Lane Detection", lane_overlay)
        cv2.imshow("Binary Lane", binary)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User exit")
    except Exception as e:
        rospy.logerr(f"Error processing frame: {e}")

if __name__ == "__main__":
    rospy.init_node(subscriberNodeName, anonymous=True)
    rospy.Subscriber(topicName, Image, callbackFunction, queue_size=1, buff_size=2**24)
    cv2.startWindowThread()
    
    t = threading.Thread(target=rospy.spin)
    t.start()
    
    try:
        while not rospy.is_shutdown():
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows() 