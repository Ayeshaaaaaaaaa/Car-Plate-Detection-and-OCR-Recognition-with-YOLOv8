from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import io
import numpy as np
import cv2
import os
import re
import requests
from easyocr import Reader
import ssl

# Disable SSL certificate verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with a list of allowed origins or "*" to allow all
    allow_credentials=True,
    allow_methods=["*"],  # Replace with a list of allowed methods or "*" to allow all
    allow_headers=["*"],  # Replace with a list of allowed headers or "*" to allow all
)

# Load the YOLO model
model = YOLO('best.pt')  # Replace with your actual model file

# Initialize EasyOCR reader
reader = Reader(['en'])

# Define the directory to save the temporary images
TEMP_IMAGE_PATH = "resized_image.jpg"
BINARIZED_IMAGE_PATH = "binarized_image.jpg"
OUTPUT_IMAGE_PATH = "output_image_with_boxes_and_text.jpg"

def resize_image_preserve_aspect_ratio(image, target_width):
    # Get the original dimensions (height, width) from the numpy array
    height, width = image.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = width / height
    
    # Calculate the new height to maintain the aspect ratio
    target_height = int(target_width / aspect_ratio)
    
    # Resize the image using OpenCV
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    return resized_image

def additional_resize_if_needed(image, min_width, min_height):
    # Get the dimensions of the resized image
    height, width = image.shape[:2]
    
    # Check if the width is less than the minimum width and height is less than the minimum height
    if width < min_width and height < min_height:
        # Calculate the target width and height for the additional resize
        target_height = min_height
        aspect_ratio = width / height
        target_width = int(target_height * aspect_ratio)
        
        # Resize the image again using OpenCV
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        return resized_image
    else:
        # Return the original image if no additional resizing is needed
        return image

def process_and_resize_image(image):
    # Ensure the image is a numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Initial resize to ensure minimum width
    initial_resized_image = resize_image_preserve_aspect_ratio(image, 332)
    
    # Perform additional resizing if needed
    final_resized_image = additional_resize_if_needed(initial_resized_image, 330, 170)
    
    return final_resized_image

def process_and_ocr(plate_region):
    # Convert the cropped region to grayscale
    plate_region_gray = np.array(plate_region.convert("L"))

    # Resize the plate region for better OCR accuracy
    plate_resized=process_and_resize_image(plate_region_gray)
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(plate_resized, (5, 5), 0)
    
    # Apply adaptive thresholding
    binarized_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2
    )
    
    # Apply morphological operations to enhance text features
    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.morphologyEx(binarized_image, cv2.MORPH_OPEN, kernel)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_CLOSE, kernel)
    # eroded_image = cv2.erode(morph_image, kernel, iterations=1)
    # contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area < 50:  # Adjust the threshold value depending on your needs
    #         cv2.drawContours(eroded_image, [contour], -1, (255,), thickness=cv2.FILLED)  # Use a tuple for the color

    # Convert the processed image back to a Pillow image for OCR
    plate_region_processed = Image.fromarray(morph_image)

    # Perform OCR on the processed image using EasyOCR
    results = reader.readtext(np.array(plate_region_processed))
    plate_texts = [result[1] for result in results]
    plate_text = " ".join(plate_texts)
    
    return plate_text.strip(), morph_image

@app.post("/upload/")
async def detect_number_plate(file: UploadFile = File(...)):
    # Open the uploaded image file
    img = Image.open(io.BytesIO(await file.read()))

    # Resize the image for YOLO model input (640x640)
    img_resized = img.resize((640, 640))

    # Convert to RGB if necessary
    if img_resized.mode in ("RGBA", "LA") or (img_resized.mode == "P" and "transparency" in img_resized.info):
        img_resized = img_resized.convert("RGB")

    # Save the resized image temporarily
    img_resized.save(TEMP_IMAGE_PATH)

    # Perform inference with the resized image
    results = model.predict(source=TEMP_IMAGE_PATH, imgsz=640)

    # Plot the image with the bounding boxes and labels
    plt.figure(figsize=(12, 8))
    plt.imshow(img_resized)  # Display the resized image
    ax = plt.gca()

    # Extracted number plate texts
    number_plate_texts = []

    # Process the inference results and draw bounding boxes
    for result in results:
        for box in result.boxes.xyxy:  # type: ignore
            x1, y1, x2, y2 = map(int, box.tolist())  # Convert to int

            # Crop the detected number plate region
            plate_region = img_resized.crop((x1, y1, x2, y2))

            # Process and perform OCR on the cropped plate region
            plate_text, morph_image = process_and_ocr(plate_region)

            # Save the binarized image
            binarized_image_path = f"binarized_plate_{x1}_{y1}.jpg"
            cv2.imwrite(binarized_image_path, morph_image)
            cleaned_text = re.sub(r'[^A-Za-z0-9]', '', plate_text)
            number_plate_texts.append(cleaned_text)

            # Draw the bounding box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Optional: Display class names and confidence
            class_id = int(result.boxes.cls[0])  # type: ignore
            class_name = result.names[class_id]
            confidence = result.boxes.conf[0]  # type: ignore
            plt.text(x1, y1, f'{class_name} {confidence:.2f}', color='red', fontsize=12, weight='bold')

            # Display the OCR-detected number plate text on the image
            plt.text(x1, y2, f'Plate: {plate_text}', color='blue', fontsize=12, weight='bold')

    plt.axis('off')

    # Convert to RGB before saving the result image
    if plt.gcf().canvas.figure.get_facecolor() == "none":
        plt.gcf().canvas.figure.set_facecolor('white')

    plt.savefig(OUTPUT_IMAGE_PATH, bbox_inches='tight', pad_inches=0)

    # Return the OCR-extracted number plate texts and the saved images
    return JSONResponse(content={
        "number_plate_texts": number_plate_texts,
        "output_image_url": f"/static/{OUTPUT_IMAGE_PATH}",
        "binarized_image_url": binarized_image_path
    })

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
