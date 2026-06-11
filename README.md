# realtime-panoptic-segmentation
# Project Description

This project implements a Real-Time Object Classification system using Panoptic Segmentation. Built with Facebook's Detectron2 and OpenCV, it utilizes a Feature Pyramid Network (FPN) with a ResNet-50 backbone to provide a complete visual understanding of a live webcam feed. 

Unlike standard bounding-box detection, this system segments both "Things" (countable, individual instances like people or cars) and "Stuff" (amorphous background regions like walls, floors, or the sky) using the COCO dataset categories. To optimize CPU performance, inference is triggered on-demand via keyboard input rather than processing every frame continuously.

## Features

* **Live Webcam Integration:** Captures high-resolution frames using OpenCV.
* **Panoptic Segmentation:** Simultaneously performs instance segmentation (Things) and semantic segmentation (Stuff).
* **On-Demand Processing:** Press a key to capture and analyze a specific frame without lagging the CPU.
* **Confidence Filtering:** Ignores weak predictions by strictly enforcing a $\geq$ 0.6 confidence threshold.
* **Dual Output:** Generates both a structured terminal summary (listing counts of detected objects) and a colorized visual mask overlay saved automatically as a `.png`.

## Prerequisites & Installation

To run this project, you will need Python installed along with the following primary libraries:

1. PyTorch
2. OpenCV (`opencv-python`)
3. Detectron2 

You will also need the pre-trained weights file (`model_final_c10459.pkl`) placed in your project directory. *(Note: This file is excluded from the repository due to size limits).*

## Usage

1. Clone the repository and ensure your webcam is connected.
2. Run the main script from your terminal:
```bash
   python main.py
