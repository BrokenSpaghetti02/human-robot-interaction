import os
import pyaudio
import google.auth
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
import sounddevice as sd
import soundfile as sf
from google.cloud import texttospeech
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
import time
from google.cloud import speech

#------------------------------------------------- Initialization and Definition -------------------------------------------------

# 设置 Google Cloud 凭据环境变量
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ubuntu/mini_pupper_bsp/demos/super_key.json"
credentials, project_id = google.auth.default()

# Initialize System and Human Message
system_message = """
You are Mini Pupper, a small robo puppy. You will be a helpful AI assistant, but your responses should always be short and direct.
But your answers will always be short and never use emojis.
"""

human_message = """
The current context is: {history}
User: {input}
"""

# Initialize Model
model = ChatVertexAI(
    model_name='gemini-pro',
    convert_system_message_to_human=True,
)

# Initialize Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """
            You are a robot puppy who is a helpful AI assistant. Your name is Mini Pupper V2.
            You must always answer in short and direct style unless necessary.
            """
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

# Set memory for context
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation = ConversationChain(llm=model, prompt=prompt, verbose=False, memory=memory)

# 创建 Speech-to-Text 客户端
speech_client = speech.SpeechClient()

# Defining function to transcribe
def transcribe_file_with_enhanced_model(path: str) -> str:
    """Transcribe the given audio file using an enhanced model."""

    with open(path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="en-US",
        use_enhanced=True,
        model="phone_call",
    )

    response = speech_client.recognize(config=config, audio=audio)
    
    for result in response.results:
        alternative = result.alternatives[0]
        return alternative.transcript

# Set the default speaker volume to maximum
os.system("amixer -c 0 sset 'Headphone' 100%")

# Create the TextToSpeechClient instance
tts_client = texttospeech.TextToSpeechClient()

print("Initialization Complete\n")
time.sleep(1)

def audio_generator(stream, chunk):
    while True:
        data = stream.read(chunk)
        if not data:
            break
        yield data

# Initialize PyAudio parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks

# 打开音频流
p = pyaudio.PyAudio()

def open_stream():
    return p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

stream = open_stream()

def detect_speech_and_transcribe():
    print("Listening...")  # 提示开始监听
    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator(stream, CHUNK))
    streaming_config = speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US"
        ),
        interim_results=True
    )
    responses = speech_client.streaming_recognize(config=streaming_config, requests=requests)

    user_input = ""
    for response in responses:
        for result in response.results:
            if result.is_final:
                if result.alternatives:
                    user_input = result.alternatives[0].transcript
                return user_input

try:
    while True:
        #------------------------------------------------- Speech to Text -------------------------------------------------
        
        # Record audio
        print("Mini Pupper 2 is listening...\n")
        user_input = detect_speech_and_transcribe()
        print(f"Mini Pupper has stopped listening. User input: {user_input}\n")

        #------------------------------------------------- Chat with Gemini Pro -------------------------------------------------

        # Fetch response from Gemini Pro
        start_time = time.time()
        result = conversation.invoke(input=user_input)['response']
        end_time = time.time()
        print(f"Gemini response time: {end_time - start_time:.2f} seconds\n")

        # Set the text to be synthesized
        text_to_speak = result

        #------------------------------------------------- Text to Speech -------------------------------------------------

        # Configure the synthesis request
        synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-GB-News-I"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        start_time = time.time()
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        end_time = time.time()
        print(f"Text to Speech processing time: {end_time - start_time:.2f} seconds\n")

        # Save the audio to a file
        with open("/home/ubuntu/mini-pupper-2-output.wav", "wb") as out:
            out.write(response.audio_content)
            print(f'Audio content written to file "mini-pupper-2-output.wav\n"')

        # Play audio
        print("Mini Pupper 2 audio playback start...\n")
        start_time = time.time()
        audio, fs = sf.read("/home/ubuntu/mini-pupper-2-output.wav")
        sd.play(audio, fs)
        sd.wait()  # Wait for playback to finish
        end_time = time.time()
        print(f"Mini Pupper 2 audio playback end. Time taken: {end_time - start_time:.2f} seconds\n")

        # Reset the stream for the next iteration
        stream.stop_stream()
        stream.close()
        stream = open_stream()

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # 关闭音频流
    print("Stopping...")  # 提示停止
    if stream.is_active():
        stream.stop_stream()
    stream.close()
    p.terminate()