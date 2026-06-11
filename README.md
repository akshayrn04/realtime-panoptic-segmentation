## Realtime Panoptic Segmentation

This project implements a Real-Time Object Classification system using Panoptic Segmentation. Built with Facebook's Detectron2 and OpenCV, it utilizes a Feature Pyramid Network (FPN) with a ResNet-50 backbone to provide a complete visual understanding of a live webcam feed.

Unlike standard bounding-box detection, this system segments both "Things" (countable, individual instances like people or cars) and "Stuff" (amorphous background regions like walls, floors, or the sky) using the COCO dataset categories. To optimize CPU performance, inference is triggered on-demand via keyboard input rather than processing every frame continuously.

### Features

* **Live Webcam Integration:** Captures high-resolution frames using OpenCV.
* **Panoptic Segmentation:** Simultaneously performs instance segmentation and semantic segmentation.
* **On-Demand Processing:** Press a key to capture and analyze a specific frame without lagging the CPU.
* **Confidence Filtering:** Ignores weak predictions by strictly enforcing a >= 0.6 confidence threshold.
* **Dual Output:** Generates both a structured terminal summary and a colorized visual mask overlay saved automatically as a .png image.

### Prerequisites and Installation

To run this project, you will need Python installed along with the following primary libraries:
* PyTorch
* OpenCV (opencv-python)
* Detectron2

You will also need the pre-trained weights file (model_final_c10459.pkl) placed in your project directory. 

### Usage

1. Clone the repository and ensure your webcam is connected.
2. Run the main script from your terminal using `python main.py`.
3. Press **C** to capture the current frame and run the AI prediction.
4. Press **Q** to quit the application.

### Output Format

* **Terminal Output:** A clean, structured list categorizing and counting the items found (e.g., Person 1, Person 2, Floors, Walls).
* **Saved Image:** A color-coded segmentation mask rendered at 1.2x scale, saved directly to the output_images folder with an auto-incrementing index.

### System Architecture

* **Framework:** Detectron2 (running on PyTorch)
* **Backbone:** ResNet-50 (R-50)
* **Neck:** Feature Pyramid Network (FPN)
* **Dataset:** COCO Panoptic 2017
* **Hardware Profile:** Configured for CPU portability

### Future Scope

* GPU acceleration for seamless, real-time video streaming.
* Upgrading to larger backbones (e.g., R-101 or Swin-T) for higher accuracy.
* Integrating a REST API and Web Dashboard for remote live stream viewing.
