# Car-Plate-Detection-and-OCR-Recognition-with-YOLOv8
This project utilizes YOLOv8 for real-time car plate detection and OCR (Optical Character Recognition) to extract plate numbers from detected regions. The system leverages advanced computer vision techniques to streamline the recognition process and provide precise results.
# Fine-Tuning YOLOv8 for License Plate Detection
In this project, I fine-tuned the YOLOv8 model to specialize in detecting license plates using a custom dataset. Starting with the pre-trained YOLOv8 model (yolov8n.pt), I adapted it to my specific use case of license plate detection by training it on a labeled dataset of license plates.

## Training Details:

Epochs: 100
Image Size: 640x640 pixels
Batch Size: 16
Learning Rate: Initial learning rate of 0.01 with gradual decay
Momentum: 0.937
Weight Decay: 0.0005
Warmup Epochs: 3
Data Augmentation: Enabled
Rectangular Training: Used to maintain aspect ratio
![image](https://github.com/user-attachments/assets/2e1d7260-28b6-4150-8fba-f264945d5d06)

## Evaluation Metrics:

Mean Precision: 0.8748 - Indicates the proportion of true positive detections among all detections.
Mean Recall: 0.8333 - Reflects the model's ability to identify all relevant license plates in the dataset.
mAP@0.5: 0.8429 - Mean Average Precision at an IoU threshold of 0.5, showing strong object localization performance.
mAP@0.5:0.95: 0.6182 - Measures model performance across multiple IoU thresholds, demonstrating robustness.
Approximate Accuracy: 0.8535 - A combined measure of precision and recall, providing a comprehensive view of model accuracy.

