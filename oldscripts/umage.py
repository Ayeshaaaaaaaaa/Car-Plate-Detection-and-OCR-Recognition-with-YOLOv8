from PIL import Image

def print_image_dimensions(image_path):
    # Open the image using the image_path parameter
    image = Image.open(image_path)

    # Get image dimensions
    width, height = image.size

    # Print dimensions
    print(f"Width: {width} pixels")
    print(f"Height: {height} pixels")

if __name__ == "__main__":
    # Call the function with the actual image file path
    print_image_dimensions('binarized_plate_238_396.jpg')
