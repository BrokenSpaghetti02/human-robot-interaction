#!/usr/bin/python
from MangDang.mini_pupper.display import Display, BehaviorState
import cv2
import time

def resize_image(input_path, output_path, width, height):
    # Read the image from the specified file
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error reading image from {input_path}")
        return False

    # Resize the image to the specified dimensions
    resized_image = cv2.resize(image, (width, height))

    # Save the resized image to the specified output path
    cv2.imwrite(output_path, resized_image)
    return True

# Define the file paths
input_image_path = 'demos/images/captured_image.jpg'
resized_image_path = 'demos/images/captured_image_resized.jpg'

# Resize the image to 320x240
if resize_image(input_image_path, resized_image_path, 320, 240):
    disp = Display()
    disp.show_image(resized_image_path)
    time.sleep(5)
    disp.show_state(BehaviorState.REST)
    time.sleep(5)
    disp.show_state(BehaviorState.TROT)
    time.sleep(5)
    disp.show_state(BehaviorState.LOWBATTERY)
    time.sleep(5)
    disp.show_ip()
else:
    print("Failed to resize the image")