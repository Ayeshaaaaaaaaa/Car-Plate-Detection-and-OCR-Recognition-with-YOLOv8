from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
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

def deskew_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to find the largest rectangle
    largest_contour = None
    largest_area = 0

    for contour in contours:
        # Approximate the contour to a polygon
        rect = cv2.minAreaRect(contour)
        area = rect[1][0] * rect[1][1]

        # Find the largest rectangle contour
        if area > largest_area:
            largest_contour = rect
            largest_area = area

    # Check if a contour was found
    if largest_contour is None:
        # If no contour was found, return the original image
        print("No contour found for deskewing.")
        return image

    # Compute the angle of the largest rectangle (deskewing angle)
    angle = largest_contour[-1]

    # Deskew the image by rotating it by the negative of the found angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def process_and_ocr(plate_region):
    # Convert the cropped region to grayscale
    plate_region_gray = np.array(plate_region.convert("L"))

    # Resize the plate region for better OCR accuracy
    plate_resized = cv2.resize(plate_region_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(plate_resized, (5, 5), 0)

    # Apply adaptive thresholding
    binarized_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
    )

    # Apply morphological operations to enhance text features
    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.morphologyEx(binarized_image, cv2.MORPH_CLOSE, kernel)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)

    # Convert the processed image back to a Pillow image for OCR
    plate_region_processed = Image.fromarray(morph_image)

    # Perform OCR on the processed image
    plate_text = pytesseract.image_to_string(plate_region_processed, config='--psm 8')

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

            # Deskew the plate region before OCR
            plate_region_cv = np.array(plate_region)
            deskewed_plate = deskew_image(plate_region_cv)

            # Convert deskewed plate back to PIL image
            plate_region_deskewed = Image.fromarray(deskewed_plate)

            # Process and perform OCR on the deskewed plate region
            plate_text, morph_image = process_and_ocr(plate_region_deskewed)

            # Save the binarized image
            binarized_image_path = f"binarized_plate_{x1}_{y1}.jpg"
            cv2.imwrite(binarized_image_path, morph_image)
            number_plate_texts.append(plate_text)

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
