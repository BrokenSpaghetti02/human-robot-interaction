import cv2
import os
from langchain_google_vertexai import ChatVertexAI
import google.auth
from vertexai.preview.generative_models import Image
from langchain_core.messages import HumanMessage, SystemMessage
import base64
from google.cloud import texttospeech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ubuntu/mini_pupper_bsp/demos/super_key.json"

credentials, project_id = google.auth.default()

model = ChatVertexAI(model="gemini-pro-vision")

def capture_image(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Camera error! Cannot turn on the camera.")
        return

    ret, frame = cap.read()

    if ret:
        # 保存图像
        cv2.imwrite(output_path, frame)
        print(f"The image was saved to {output_path}")
    else:
        print("Cannot capture the image")

    cap.release()

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
    capture_image(output_path)

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
        "text": "Who is that? Name only.",
    }

    # 准备模型输入
    message = HumanMessage(content=[text_message, image_message])

    # 调用模型
    output = model.invoke([message])
    print(output.content)

    # 生成并播放语音
    result_text = f"The person in the image is {output.content.strip()}."
    print(result_text)

    # 将结果转换为语音
    output_audio_path = "/home/ubuntu/mini_pupper_bsp/demos/result.mp3"
    text_to_speech(result_text, output_audio_path)

    # 播放音频文件
    os.system(f"mpg321 {output_audio_path}")