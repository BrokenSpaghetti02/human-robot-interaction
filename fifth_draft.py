import os
import pyaudio
import google.auth
import json
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

# Get credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ubuntu/mini_pupper_bsp/demos/super_key.json"
credentials, project_id = google.auth.default()

# Initialize model
model = ChatVertexAI(
    model_name='gemini-pro',
    convert_system_message_to_human=True,
)

# Initialize prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """
            You are Jenny, a small female robo puppy. You will be a helpful AI assistant.
            But your answers will always be short and never use emojis, hashtags, or asterisks.
            Only answer in sentences.
            """
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

# Load conversation history from file
def load_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

# Save conversation history to file
def save_history(file_path, history):
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4)

# Print conversation history
def print_history(file_path):
    history = load_history(file_path)
    for message in history:
        role = "User" if message['role'] == 'user' else "AI"
        print(f"{role}: {message['content']}")

# Define the file path for storing conversation history
history_file_path = "/home/ubuntu/conversation_history.json"

# Load the conversation history
history = load_history(history_file_path)

# Set memory for context
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Populate memory with loaded history
for message in history:
    if message['role'] == 'user':
        memory.chat_memory.add_user_message(message['content'])
    elif message['role'] == 'ai':
        memory.chat_memory.add_ai_message(message['content'])

conversation = ConversationChain(llm=model, prompt=prompt, verbose=False, memory=memory)

# Print the conversation history
print("Conversation History:\n")
print_history(history_file_path)

# Create the Speech-to-Text instance
speech_client = speech.SpeechClient()

# Set the default speaker volume to maximum
os.system("amixer -c 0 sset 'Headphone' 100%")

# Create the TextToSpeechClient instance
tts_client = texttospeech.TextToSpeechClient()

def audio_generator(stream, chunk):
    while True:
        data = stream.read(chunk)
        if not data:
            break
        yield data

# Initialize PyAudio parameters
RATE = 48000
CHUNK = int(RATE / 10)  # 100ms chunks

# Create the PyAudio instance
p = pyaudio.PyAudio()

def open_stream():
    return p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

def detect_speech_and_transcribe():
    print("Listening...")  # start listenning
    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator(stream, CHUNK))
    streaming_config = speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            use_enhanced=True,
            model="phone_call",
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

print("Initialization Complete\n")
time.sleep(1)

speak = str(input("Speak to Mini Pupper? (y/n)\n"))

while speak == 'n':
    speak = str(input("Speak to Mini Pupper? (y/n)\n"))

time.sleep(1)

stream = open_stream()

while True:

#------------------------------------------------- Speech to Text -------------------------------------------------

    # Record audio
    print("Mini Pupper 2 is listening...\n")
    user_input = detect_speech_and_transcribe()
    print(f"Mini Pupper has stopped listening. User input: {user_input}\n")

#------------------------------------------------- Chat with Gemini Pro -------------------------------------------------

    # Add the suffix to user input
    user_input_with_suffix = user_input + " In short. NO asterisks, NO hashtags, or emojis. NO asterisks, NO hashtags, or emojis. "

    # Fetch response from Gemini Pro
    start_time = time.time()
    result = conversation.invoke(input=user_input_with_suffix)['response']
    end_time = time.time()
    print(f"Gemini response time: {end_time - start_time:.2f} seconds\n")

    print(f"Mini Pupper: {result}\n")

    # Update and save conversation history
    memory.chat_memory.add_user_message(user_input_with_suffix)
    memory.chat_memory.add_ai_message(result)

    # Generate the history to be saved
    history = []
    messages = memory.chat_memory.messages
    for i in range(0, len(messages), 2):
        if i < len(messages):
            history.append({'role': 'user', 'content': messages[i].content})
        if i + 1 < len(messages):
            history.append({'role': 'ai', 'content': messages[i + 1].content})

    # Save the updated history
    save_history(history_file_path, history)

    # Set the text to be synthesized
    text_to_speak = result

#------------------------------------------------- Text to Speech -------------------------------------------------

    # Configure the synthesis request
    synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-GB",
        name="en-GB-Wavenet-F"
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

# 关闭音频流
print("Stopping...")  # 提示停止
if stream.is_active():
    stream.stop_stream()
stream.close()
p.terminate()
