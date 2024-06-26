# I am playing rock paper scissors. Tell me what is this symbol?
import cv2
import os
import time
from langchain_google_vertexai import ChatVertexAI
import google.auth
from vertexai.preview.generative_models import Image
from langchain_core.messages import HumanMessage, SystemMessage
import base64

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ubuntu/mini_pupper_bsp/demos/super_key.json"

credentials, project_id = google.auth.default()

model = ChatVertexAI(model="gemini-pro-vision")

def capture_image(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # turn on the camera
    cap = cv2.VideoCapture(0)  # 0 represents the "first" camera
    if not cap.isOpened():
        print("Camera error! Cannot turn on the camera.")
        return

    # read one frame
    ret, frame = cap.read()

    if ret:
        # save the image
        cv2.imwrite(output_path, frame)
        print(f"The image was saved to {output_path}")
    else:
        print("Cannot capture the image")

    # release the camera
    cap.release()

if __name__ == "__main__":
    # Measure total time
    start_time_total = time.time()

    output_path = "/home/ubuntu/mini_pupper_bsp/demos/images/captured_image.jpg"

    # Measure time for capturing the image
    start_time_capture = time.time()
    capture_image(output_path)
    end_time_capture = time.time()
    capture_duration = end_time_capture - start_time_capture
    print(f"Time taken to capture and save the image: {capture_duration:.2f} seconds")

    with open(output_path, "rb") as image_file:
        image_bytes = image_file.read()

    image_message = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
        },
    }
    text_message = {
        "type": "text",
        "text": "I am playing rock paper scissors. Tell me what is this symbol? Only in one word.",
    }

    # Prepare input for model consumption
    message = HumanMessage(content=[text_message, image_message])

    # Measure time for model invocation
    start_time_model = time.time()
    output = model.invoke([message])
    end_time_model = time.time()
    model_duration = end_time_model - start_time_model
    print(f"Time taken for model invocation and response: {model_duration:.2f} seconds")

    print("Model response:", output.content)

    # Measure total time
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    print(f"Total time taken: {total_duration:.2f} seconds")