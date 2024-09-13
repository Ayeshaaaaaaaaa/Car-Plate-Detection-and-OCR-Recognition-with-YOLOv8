from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import pytesseract
import io
import numpy as np
import cv2
import os

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

# Define the directory to save the temporary images
TEMP_IMAGE_PATH = "resized_image.jpg"
BINARIZED_IMAGE_PATH = "binarized_image.jpg"
OUTPUT_IMAGE_PATH = "output_image_with_boxes_and_text.jpg"

@app.post("/upload/")
async def detect_number_plate(file: UploadFile = File(...)):
    # Open the uploaded image file
    img = Image.open(io.BytesIO(await file.read()))

    # Resize the image for YOLO model input (640x640)
    img_resized = img.resize((640, 640))

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

            # Convert the cropped region to grayscale
            plate_region_gray = plate_region.convert("L")

            # Convert PIL image to NumPy array
            plate_region_np = np.array(plate_region_gray)

            # Apply adaptive thresholding
            binary_image = cv2.adaptiveThreshold(plate_region_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Save the binarized image
            cv2.imwrite(BINARIZED_IMAGE_PATH, binary_image)

            # Convert binary NumPy array back to PIL image
            plate_region_binary = Image.fromarray(binary_image)

            # Perform OCR on the binarized image
            plate_text = pytesseract.image_to_string(plate_region_binary, config='--psm 8')  # Single word mode
            number_plate_texts.append(plate_text.strip())

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
            plt.text(x1, y2, f'Plate: {plate_text.strip()}', color='blue', fontsize=12, weight='bold')

    plt.axis('off')

    # Save the result with bounding boxes and number plate text as a new image
    plt.savefig(OUTPUT_IMAGE_PATH, bbox_inches='tight', pad_inches=0)

    # Return the OCR-extracted number plate texts and the saved images
    return {
        "number_plate_texts": number_plate_texts,
        "output_image": FileResponse(OUTPUT_IMAGE_PATH, media_type='image/jpeg'),
        "binarized_image": FileResponse(BINARIZED_IMAGE_PATH, media_type='image/jpeg')
    }

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
