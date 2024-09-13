from PIL import Image
import io


def print_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Width: {width}, Height: {height}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print_image_dimensions('binarized_plate_59_136.jpg')

