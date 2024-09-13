from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import pytesseract
import io
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

# Define the directory to save the temporary resized image
TEMP_IMAGE_PATH = "resized_image.jpg"

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
            x1, y1, x2, y2 = box.tolist()

            # Crop the detected number plate region
            plate_region = img_resized.crop((x1, y1, x2, y2))

            # Perform OCR on the cropped region
            plate_text = pytesseract.image_to_string(plate_region, config='--psm 8')  # Single word mode
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
    output_image_path = "output_image_with_boxes_and_text.jpg"
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)

    # Return the OCR-extracted number plate texts and the saved image
    return {
        "number_plate_texts": number_plate_texts,
        "image": FileResponse(output_image_path, media_type='image/jpeg')
    }

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
