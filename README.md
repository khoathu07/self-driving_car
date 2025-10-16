Autonomous Vehicle Using Computer Vision and AI
This project develops a self-driving car model capable of autonomous navigation in a simulated environment. The system uses a DC motor for speed control, a Servo motor for steering, and a front camera for real-time image processing. Techniques like color thresholding, binary conversion, and sliding window are applied for lane detection, while the Pure Pursuit algorithm handles steering control.

![alt](https://github.com/user-attachments/assets/5b5b9ff1-d41c-444d-aa34-8b96dc171d1a)

The vehicle recognizes key traffic signs (“Stop”, “Turn Left”, “Turn Right”, “Parking”) with about 95% training accuracy and executes corresponding actions effectively. Deployed on Jetson Nano, the system achieves real-time lane tracking and decision-making.

![alt](https://github.com/user-attachments/assets/99e7d1d8-dfc3-4ee2-8288-d152a5c0f864)

Highlights: Accurate lane following, reliable sign recognition, and stable performance.
Future work: Integrate deep learning models (e.g., LaneNet) to improve lane detection and enhance embedded hardware for higher FPS and real-time efficiency.