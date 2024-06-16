# I am playing rock paper scissors. Tell me what is this symbol?
import time
import os
import random
import base64
import importlib
import cv2
import sys
from google.cloud import texttospeech

# Measure total time from the very beginning
start_time_total = time.time()

# Dictionary to store import times
import_times = {}

def measure_import_time(module_name):
    start_time = time.time()
    globals()[module_name] = importlib.import_module(module_name)
    end_time = time.time()
    import_times[module_name] = end_time - start_time

# Measure time for each import
measure_import_time('cv2')
measure_import_time('os')
measure_import_time('time')
measure_import_time('random')
measure_import_time('base64')

# For specific imports
start_time = time.time()
from langchain_google_vertexai import ChatVertexAI
end_time = time.time()
import_times['ChatVertexAI'] = end_time - start_time

start_time = time.time()
import google.auth
end_time = time.time()
import_times['google.auth'] = end_time - start_time

start_time = time.time()
from vertexai.preview.generative_models import Image
end_time = time.time()
import_times['Image'] = end_time - start_time

start_time = time.time()
from langchain_core.messages import HumanMessage, SystemMessage
end_time = time.time()
import_times['HumanMessage'] = end_time - start_time

# Print import times
for module, duration in import_times.items():
    print(f"Time taken to import {module}: {duration:.2f} seconds")

# Set up Google Cloud credentials for local environment
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ubuntu/mini_pupper_bsp/demos/super_key.json"

credentials, project_id = google.auth.default()

model = ChatVertexAI(model="gemini-pro-vision")

end_time_initial_setup = time.time()
initial_setup_duration = end_time_initial_setup - start_time_total
print(f"Time taken for initial setup (imports and initialization): {initial_setup_duration:.2f} seconds")

cap = cv2.VideoCapture(0)  # 0 represents the "first" camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

def capture_image(output_path, cap):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # read one frame
    ret, frame = cap.read()

    if ret:
        # save the image
        cv2.imwrite(output_path, frame)
        print(f"The image was saved to {output_path}")
    else:
        print("Cannot capture the image")

def determine_winner(user_gesture, ai_gesture):
    if user_gesture == ai_gesture:
        return "It's a tie!"
    elif (user_gesture == "rock" and ai_gesture == "scissors") or \
         (user_gesture == "scissors" and ai_gesture == "paper") or \
         (user_gesture == "paper" and ai_gesture == "rock"):
        return "You win!"
    else:
        return "You lose!"

def text_to_speech(text, output_audio_path):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-GB",
        name="en-GB-Wavenet-F"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)
    print(f"Audio content written to file '{output_audio_path}'")

if __name__ == "__main__":
    output_path = "/home/ubuntu/mini_pupper_bsp/demos/images/captured_image.jpg"

    # Measure time for capturing the image
    start_time_capture = time.time()
    capture_image(output_path, cap)
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
        "text": "I am playing rock paper scissors. Tell me what is this symbol? Only in one word, no punctuation and all in lowercase.",
    }

    # Prepare input for model consumption
    message = HumanMessage(content=[text_message, image_message])

    # Measure time for model invocation
    start_time_model = time.time()
    output = model.invoke([message])
    end_time_model = time.time()
    model_duration = end_time_model - start_time_model
    print(f"Time taken for model invocation and response: {model_duration:.2f} seconds")

    user_gesture = output.content.strip().lower()
    print(f"Model response: {user_gesture}")

    # Randomly generate a gesture for the AI
    gestures = ["rock", "paper", "scissors"]
    ai_gesture = random.choice(gestures)
    print(f"AI's gesture: {ai_gesture}")

    # Determine the winner
    result = determine_winner(user_gesture, ai_gesture)
    print(f"Your gesture: {user_gesture}, AI's gesture: {ai_gesture}. {result}")

    # Generate result text for TTS
    result_text = f"My choice is {ai_gesture} and your choice is {user_gesture}. {result}."
    print(result_text)

    # Convert result text to speech
    output_audio_path = "/home/ubuntu/mini_pupper_bsp/demos/result.mp3"
    text_to_speech(result_text, output_audio_path)

    # Play the audio file
    os.system(f"mpg321 {output_audio_path}")

    # Measure total time
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    print(f"Total time taken: {total_duration:.2f} seconds")

    # Release the camera
    cap.release()