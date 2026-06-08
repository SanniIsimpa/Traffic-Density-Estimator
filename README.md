# Traffic Density Estimator

A lightweight, real-time Computer Vision application designed to monitor traffic flow and estimate density. This system leverages deep learning to detect and track vehicles, providing actionable insights for urban traffic management.

## 🚀 The "Why"
Urban traffic congestion is a persistent challenge. This project provides a scalable, edge-compatible solution for real-time traffic monitoring. By processing video feeds locally using optimized deep learning models, this system delivers immediate congestion metrics without requiring expensive cloud infrastructure or high-bandwidth data transmission.

## 🛠️ The Challenge & Solution
During the development phase, I addressed two core engineering challenges:

* **Challenge 1: High Latency on CPU.** The initial implementation struggled to maintain real-time processing speeds on standard consumer hardware.
    * **Solution:** I optimized the pipeline by implementing the `yolov8n` (Nano) architecture. This provided the optimal speed-to-accuracy trade-off, allowing for high-performance inference on CPU-bound devices without sacrificing essential detection reliability.
* **Challenge 2: Identity Switching.** In dense traffic environments, standard trackers often failed to maintain object consistency, leading to inaccurate unique vehicle counts.
    * **Solution:** I integrated the ByteTrack algorithm, which is highly robust against occlusion. This ensures that individual vehicle identities remain persistent even when objects overlap or briefly disappear from the camera’s field of view.

## ⚙️ Key Technical Features
* **Real-time Performance Monitoring:** The HUD tracks and displays the current **Frames Per Second (FPS)**, allowing for benchmarking and performance tuning on varying hardware configurations.
* **Configurable Density Logic:** All threshold settings are centralized. The system can be calibrated to define "Low," "Moderate," or "Heavy" traffic based on the specific requirements of the deployment location and camera angle.
* **Model Flexibility:** The modular code structure allows for seamless swapping of model variants (e.g., upgrading from `n` to `s` or `m` models) if higher precision is required for specific high-stakes environments.



## 🛠️ Installation
1. Ensure you have Python 3.8+ installed.
2. Clone this repository:
   ```bash
   git clone [https://github.com/SanniIsimpa/PresenceMonitor.git](https://github.com/SanniIsimpa/PresenceMonitor.git)