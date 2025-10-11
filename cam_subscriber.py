#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import threading

subscriberNodeName = 'camera_sensor_subscriber'
topicName = 'video_topic'

bridge = CvBridge()
previous_time = time.time()

# Cache filters và transform matrix
gaussian_filter = None
canny_detector = None

def init_cuda_filters(gpu_gray):
    global gaussian_filter, canny_detector
    if gaussian_filter is None:
        gaussian_filter = cv2.cuda.createGaussianFilter(
            gpu_gray.type(), gpu_gray.type(), (5,5), 0)
    if canny_detector is None:
        canny_detector = cv2.cuda.createCannyEdgeDetector(150, 255)

def phang_hoa(img_cpu):
    """
    WarpPerspective hoàn toàn trên GPU.
    Trả về: GpuMat warped image
    """
    wT, hT = 640, 480
    src = np.float32([[154,215],[wT-154,215],[0,428],[wT,428]])
    dst = np.float32([[0,0],[wT,0],[0,hT],[wT,hT]])
    
    # Tính và cache ma trận M (numpy)
    if not hasattr(phang_hoa, "_M_cpu"):
        phang_hoa._M_cpu = cv2.getPerspectiveTransform(src, dst).astype(np.float32)
    
    # Upload ảnh CPU lên GPU
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img_cpu)
    
    # WarpPerspective trên GPU, dùng M numpy
    gpu_warped = cv2.cuda.warpPerspective(
        gpu_img,
        phang_hoa._M_cpu,
        (wT, hT),
        flags=cv2.INTER_LINEAR
    )
    return gpu_warped

def process_image(gpu_rect):
    """
    Tiếp nhận GpuMat (warped BGR), xử lý:
     - convert to gray
     - gaussian blur
     - canny
    Trả về hai GpuMat: gray và edges
    """
    global gaussian_filter, canny_detector
    gpu_gray = cv2.cuda.cvtColor(gpu_rect, cv2.COLOR_BGR2GRAY)
    init_cuda_filters(gpu_gray)
    gpu_blur  = gaussian_filter.apply(gpu_gray)
    gpu_edges = canny_detector.detect(gpu_blur)
    return gpu_gray, gpu_edges

def callbackFunction(msg):
    global previous_time
    rospy.loginfo("Received a video frame")
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        gpu_rect = phang_hoa(frame)
        gpu_gray, gpu_edges = process_image(gpu_rect)
        
        rectified = gpu_rect.download()
        edges     = gpu_edges.download()
        
        now = time.time()
        fps = 1.0 / (now - previous_time)
        previous_time = now
        
        cv2.putText(rectified, f"FPS: {fps:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow("Phang Hoa (CUDA)", rectified)
        cv2.imshow("Canny Edges (CUDA)", edges)
        
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
