import cv2
import os

def capture_image(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # turn on the camera
    cap = cv2.VideoCapture(0)  # 0 represent the "first" camera
    if not cap.isOpened():
        print("Camera error! Cannot turn on the camera.")
        return

    # read one frame
    ret, frame = cap.read()

    if ret:
        # save the image
        cv2.imwrite(output_path, frame)
        print(f"the image was saved to {output_path}")
    else:
        print("Cannot capture the iamge")

    # release the camera
    cap.release()

if __name__ == "__main__":

    output_path = "/home/ubuntu/mini_pupper_bsp/demos/images/captured_image.jpg"
    capture_image(output_path)