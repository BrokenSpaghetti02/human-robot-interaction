#!/usr/bin/python
from MangDang.mini_pupper.display import Display
import cv2
import time

def play_video(video_path, width, height):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return False

    disp = Display()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the specified dimensions
        resized_frame = cv2.resize(frame, (width, height))

        # Save the resized frame to a temporary file
        temp_frame_path = '/tmp/temp_frame.jpg'
        cv2.imwrite(temp_frame_path, resized_frame)

        # Display the frame
        disp.show_image(temp_frame_path)
        time.sleep(1 / 30)  # Adjust the delay to match the video's frame rate

    cap.release()
    return True

if __name__ == "__main__":
    # Define the file path for the video
    input_video_path = '/home/ubuntu/mini_pupper_bsp/demos/images/captured_video.mp4'  # Path to your video file

    # Play the video
    if not play_video(input_video_path, 320, 240):
        print("Failed to play the video")