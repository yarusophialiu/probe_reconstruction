import numpy as np
# import pyexr
from PIL import Image, ImageDraw

def create_distance_map(width, height, center_x, center_y):
    # Create a blank image
    image = Image.new("L", (width, height), color=0)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Draw a filled white circle at the specified center
    draw.ellipse((center_x - 50, center_y - 50, center_x + 50, center_y + 50), fill=255)

    # Convert the image to a NumPy array with float32 dtype
    distance_map = np.array(image, dtype=np.float32)

    return distance_map


if __name__ == "__main__":
    # image_float32 = np.random.rand(512, 512, 3).astype(np.float32)
    #
    # # Save the image in OpenEXR format

    # pyexr.write("output_image.exr", image_float32)

    # Specify image dimensions and center coordinates
    width, height = 512, 512
    center_x, center_y = width // 2, height // 2

    # Create a distance map
    distance_map = create_distance_map(width, height, center_x, center_y)

    # Save the distance map as an image
    image = Image.fromarray(distance_map)
    image.save("distance_map.png")