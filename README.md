# Car-Plate-Detection-and-OCR-Recognition-with-YOLOv8
This project utilizes YOLOv8 for real-time car plate detection and OCR (Optical Character Recognition) to extract plate numbers from detected regions. The system leverages advanced computer vision techniques to streamline the recognition process and provide precise results.
# Fine-Tuning YOLOv8 for License Plate Detection
In this project, I fine-tuned the YOLOv8 model to specialize in detecting license plates using a custom dataset. Starting with the pre-trained YOLOv8 model (yolov8n.pt), I adapted it to my specific use case of license plate detection by training it on a labeled dataset of license plates.

## Training Details:

Epochs: 100<br>
Image Size: 640x640 pixels<br>
Batch Size: 16<br>
Learning Rate: Initial learning rate of 0.01 with gradual decay<br>
Momentum: 0.937<br>
Weight Decay: 0.0005<br>
Warmup Epochs: 3<br>
Data Augmentation: Enabled<br>
Rectangular Training: Used to maintain aspect ratio<br>
![image](https://github.com/user-attachments/assets/2e1d7260-28b6-4150-8fba-f264945d5d06)

## Evaluation Metrics:

Mean Precision: 0.8748 - Indicates the proportion of true positive detections among all detections.<br>
Mean Recall: 0.8333 - Reflects the model's ability to identify all relevant license plates in the dataset.<br>
mAP@0.5: 0.8429 - Mean Average Precision at an IoU threshold of 0.5, showing strong object localization performance.<br>
mAP@0.5:0.95: 0.6182 - Measures model performance across multiple IoU thresholds, demonstrating robustness.<br>
Approximate Accuracy: 0.8535 - A combined measure of precision and recall, providing a comprehensive view of model accuracy.<br>

## Loading YOLOv8 and EasyOCR Models:
The YOLOv8 model is loaded for detecting car plates, and the EasyOCR library is initialized to read text from detected plates.<br>
The YOLO('best.pt') command loads the pre-trained YOLOv8 model from the specified file. The Reader(['en']) initializes the EasyOCR model to recognize English text.<br>
![image](https://github.com/user-attachments/assets/0f2e0884-25b6-44b9-86ca-73201d1e1881)
<br>
## Handling Image Upload and Processing:
When a user uploads an image, it is resized to 640x640 (the required size for YOLO), and plate detection is performed.<br>

![image](https://github.com/user-attachments/assets/c22fa395-f8bd-415d-ad43-70aa855ff411)
## License Plate Detection and OCR Processing:
Once YOLO detects bounding boxes (plates), the cropped region is processed using OCR.<br>

![image](https://github.com/user-attachments/assets/b8526ee7-7330-4e5d-ac29-5437b9cd3ae7)
## Image Processing for Better OCR:
The process_and_ocr function enhances the OCR accuracy by converting the cropped image to grayscale, resizing it, applying Gaussian blur, and using adaptive thresholding to binarize the image.<br>

![image](https://github.com/user-attachments/assets/ea76680a-bbc3-4953-a8f4-2b07dd51fade)
## Displaying Results:
Bounding boxes and detected plate numbers are drawn on the original image for visualization.<br>

![image](https://github.com/user-attachments/assets/059d7ce9-31d4-4419-888d-bee1f7030253)
## Returning the Results:
The extracted plate numbers and paths to saved images (with bounding boxes and processed plates) are returned as a JSON response.<br>

![image](https://github.com/user-attachments/assets/b3d5c21d-d9e3-436b-895f-843bec6c8a84)
## Contribution and Collaboration
I warmly welcome contributions and collaboration on this project! Whether you have suggestions for improvements, new features, or bug fixes, your input is highly valued. To contribute, please follow these steps:<br>

Fork the Repository: Create a personal copy of the repository to work on.<br>
Create a Branch: Develop your changes on a separate branch to keep the main branch stable.<br>
Make Changes: Implement your enhancements or fixes and test them thoroughly.<br>
Submit a Pull Request: Once you’re satisfied with your changes, submit a pull request for review.<br>
