import json
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
import importlib
import subprocess

# Function to get credentials
def get_credentials():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ubuntu/mini_pupper_bsp/demos/super_key.json"
    credentials, project_id = google.auth.default()

# Function to create all the instances
def create_instances(RATE, CHUNK):
    # Create model instance
    model = ChatVertexAI(
        model_name='gemini-pro',
        convert_system_message_to_human=True,
    )

    # Create prompt instance
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are Jenny, a small female robo puppy. You will be a helpful AI assistant.
                ---
                (Sentences in this bracket are your instructions which you MUST FOLLOW but NEVER UTTER EVER.
                NEVER USE emojis, hashtags, or asterisks.
                ONLY answer in VERY SHORT sentences.
                Please detect the following tasks the user intentends you to do. ONLY reply with 'INTENT_PLAY_GAME' if 
                the user intends to play a rock paper scissors game. ONLY reply with 'INTENT_RECOGNIZE_FACE' if the user 
                intends you to recognize a face. ONLY reply with 'INTENT_MOVE' if the user intends you to move or go in any 
                direction. Otherwise, start with 'NO_INTENT'.)
                """
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    # Create memory instances
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # Create Speech-to-Text instance
    speech_client = speech.SpeechClient()

    # Create TextToSpeechClient instance
    tts_client = texttospeech.TextToSpeechClient()

    # Create PyAudio instance
    p = pyaudio.PyAudio()

    # Create voice instance
    voice = texttospeech.VoiceSelectionParams(
            language_code="en-GB",
            name="en-GB-Wavenet-F"
        )
    
    # Create audio configuration instance
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    return model, prompt, memory, speech_client, tts_client, p, voice, audio_config

def open_stream(RATE, CHUNK, p):
    return p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

def audio_generator(stream, chunk):
    try:
        while True:
            data = stream.read(chunk)
            if not data:
                # End of stream, break out of the loop
                break
            yield data
    except Exception as e:
        # Handle any exceptions that may occur while reading from the stream
        print(f"Error reading from audio stream: {e}")
        # You may want to close the stream or return a signal to indicate an error
        yield None

# Function to detect voice and transribe speech
def detect_speech_and_transcribe(stream, RATE, CHUNK, speech_client):
    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator(stream = stream, chunk = CHUNK))
    streaming_config = speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz = RATE,
            language_code = "en-US",
            use_enhanced = True,
            model="phone_call",
        ),
        interim_results=True
    )
    responses = speech_client.streaming_recognize(config = streaming_config, requests = requests)

    user_input = ""
    for response in responses:
        for result in response.results:
            if result.is_final:
                if result.alternatives:
                    user_input = result.alternatives[0].transcript
                return user_input

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

def main():
    get_credentials()

    # Set the default speaker volume to maximum
    os.system("amixer -c 0 sset 'Headphone' 100%")
    os.system('cls' if os.name == 'nt' else 'clear')

    RATE = 48000
    CHUNK = int(48000 / 10)
    model, prompt, memory, speech_client, tts_client, p, voice, audio_config = create_instances(RATE = RATE, CHUNK = CHUNK)

    # Define the file path for storing conversation history
    history_file_path = "/home/ubuntu/conversation_history.json"

    # Load the conversation history
    history = load_history(history_file_path)

    # Populate memory with loaded history
    for message in history:
        if message['role'] == 'user':
            memory.chat_memory.add_user_message(message['content'])
        elif message['role'] == 'ai':
            memory.chat_memory.add_ai_message(message['content'])

    conversation = ConversationChain(llm=model, prompt=prompt, verbose=False, memory=memory)

    print("Initialization Complete!\n")
    time.sleep(0.5)

    speak = str(input("Speak to Mini Pupper? (y/n)\n"))

    while speak == 'n':
        speak = str(input("Speak to Mini Pupper? (y/n)\n"))

    time.sleep(0.5)

    while True:

#------------------------------------------------- Speech to Text -------------------------------------------------

        print("Mini Pupper 2 is listening...\n")
        stream = open_stream(RATE = RATE, CHUNK = CHUNK, p = p)
        user_input = detect_speech_and_transcribe(stream = stream, RATE = RATE, CHUNK = CHUNK, speech_client = speech_client)
        print(f"Mini Pupper has stopped listening.\n")
        print(f"User input: {user_input}\n")

        while user_input == "":
            speak = str(input("No user input. Speak to Mini Pupper? (y/n)\n"))

            while speak == 'n':
                speak = str(input("No user input. Speak to Mini Pupper? (y/n)\n"))

            stream.stop_stream()
            stream.close()
            print("Mini Pupper 2 is listening...\n")
            stream = open_stream(RATE = RATE, CHUNK = CHUNK, p = p)
            user_input = detect_speech_and_transcribe(stream = stream, RATE = RATE, CHUNK = CHUNK, speech_client = speech_client)
            print(f"Mini Pupper has stopped listening.\n")
            print(f"User input: {user_input}\n")

#------------------------------------------------- Chat with Gemini Pro with Intention Detection -------------------------------------------------

        # Add intention detection prompt
        combined_input = user_input

        # Fetch response from Gemini Pro
        start_time = time.time()
        result = conversation.invoke(input=combined_input)['response']
        end_time = time.time()
        print(f"Gemini response time: {end_time - start_time:.2f} seconds\n")

        print(f"Debug: Gemini response: {result}\n")  # Debug line to print the full response

        # Check intention detection result
        if 'INTENT_PLAY_GAME' in result:
            print("User has intention to play a game. Starting the game...\n")
            subprocess.run(['python3', '/home/ubuntu/mini_pupper_bsp/demos/Game.py'])
        elif 'INTENT_RECOGNIZE_FACE' in result:
            print("User has intention to recognize a face. Starting face recognition...\n")
            subprocess.run(['python3', '/home/ubuntu/mini_pupper_bsp/demos/image_detection.py'])
        elif 'INTENT_MOVE' in result:
            print("User has intention to make me move. Moving...\n")
            subprocess.run(['python3', '/home/ubuntu/StanfordQuadruped.0603//run_movement.py'])

        # Filter out the detection result from the response
        filtered_result = result.replace('INTENT_PLAY_GAME', '').replace('INTENT_RECOGNIZE_FACE', '').replace('INTENT_MOVE', '').replace('NO_INTENT', '').strip()

        # Update and save conversation history
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(filtered_result)

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

        def remove_special_chars(text):
            return text.replace('#', '').replace('*', '').replace('`', '')
        
        filtered_result = remove_special_chars(filtered_result)

        # Set the text to be synthesized
        text_to_speak = filtered_result

#------------------------------------------------- Text to Speech -------------------------------------------------

        # Configure the synthesis request
        synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)

        start_time = time.time()
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        end_time = time.time()
        print(f"Text to Speech processing time: {end_time - start_time:.2f} seconds\n")

        # Save the audio to a file
        with open("/home/ubuntu/mini-pupper-2-output.wav", "wb") as out:
            out.write(response.audio_content)
            print(f'Audio content written to file "mini-pupper-2-output.wav"\n')

        # Play audio
        print("Mini Pupper 2 audio playback start...\n", end = '\r')
        start_time = time.time()
        audio, fs = sf.read("/home/ubuntu/mini-pupper-2-output.wav")
        sd.play(audio, fs)
        sd.wait()  # Wait for playback to finish
        end_time = time.time()
        print(f"Mini Pupper 2 audio playback end. Time taken: {end_time - start_time:.2f} seconds\n")

        # Reset the stream for the next iteration
        stream.stop_stream()
        stream.close()

if __name__ == '__main__':
    main()
