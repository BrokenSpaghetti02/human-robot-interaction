import cv2
import os
import threading

# Function to check for user input
def wait_for_stop():
    input("Press 'y' and Enter to stop recording.\n")
    global stop_recording
    stop_recording = True

def capture_video(output_path):
    global stop_recording
    stop_recording = False

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open the camera using V4L2 backend
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Cannot open the camera")
        return

    # Get the default frames per second (fps) and resolution of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # Fallback if FPS is not detected
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:  # Fallback if resolution is not detected
        width, height = 640, 480

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Recording video at {width}x{height} at {fps} FPS. Press 'y' and Enter to stop recording.")

    # Start a thread to wait for user input
    input_thread = threading.Thread(target=wait_for_stop)
    input_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Write the current frame to the video file
        out.write(frame)

        # Check if the user pressed the 'y' key
        if stop_recording:
            print("Stopping the recording.")
            break

    # Release the camera and video file
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_path = "/home/ubuntu/mini_pupper_bsp/demos/images/captured_video.mp4"  # Set the video save path
    capture_video(output_path)