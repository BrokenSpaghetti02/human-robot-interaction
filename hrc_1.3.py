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
                You are Megamind, a small male robo puppy. You will be a helpful AI assistant.
                ---
                NEVER USE emojis, hashtags, or asterisks.
                ONLY answer in SHORT sentences.
                """
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    # Create memory instances
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    conversation = ConversationChain(llm=model, prompt=prompt, verbose=False, memory=memory)

    # Create Speech-to-Text instance
    speech_client = speech.SpeechClient()

    # Create TextToSpeechClient instance
    tts_client = texttospeech.TextToSpeechClient()

    # Create PyAudio instance
    p = pyaudio.PyAudio()

    # Create voice instance
    voice = texttospeech.VoiceSelectionParams(
            language_code="en-GB",
            name="en-GB-Wavenet-B"
        )
    
    # Create audio configuration instance
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    return conversation, speech_client, tts_client, p, voice, audio_config

def open_stream(RATE, CHUNK, p):
    return p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

def audio_generator(stream, chunk):
    while True:
        data = stream.read(chunk)
        if not data:
            break
        yield data

# Function to detect voice and transribe speech
def detect_speech_and_transcribe(stream, RATE, CHUNK, speech_client):
    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator(stream = stream, chunk = CHUNK))
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

def main():
    get_credentials()
    RATE = 48000
    CHUNK = int(48000 / 10)
    conversation, speech_client, tts_client, p, voice, audio_config = create_instances(RATE = RATE, CHUNK = CHUNK)

    # Set the default speaker volume to maximum
    os.system("amixer -c 0 sset 'Headphone' 100%")
    os.system('cls' if os.name == 'nt' else 'clear')

    print("Initialization Complete!\n")
    time.sleep(0.5)

    speak = str(input("Speak to Mini Pupper? (y/n)\n"))

    while speak == 'n':
        speak = str(input("Speak to Mini Pupper? (y/n)\n"))

    time.sleep(0.5)

    stream = open_stream(RATE = RATE, CHUNK = CHUNK, p = p)

    while True:

#------------------------------------------------- Speech to Text -------------------------------------------------

        print("Mini Pupper 2 is listening...\n")
        user_input = detect_speech_and_transcribe(stream=stream, RATE = RATE, CHUNK = CHUNK, speech_client = speech_client)
        print(f"Mini Pupper has stopped listening.\n")
        print(f"User input: {user_input}\n")

        while user_input == "":
            speak = str(input("No user input. Speak to Mini Pupper? (y/n)\n"))

            while speak == 'n':
                speak = str(input("No user input. Speak to Mini Pupper? (y/n)\n"))

            print("Mini Pupper 2 is listening...\n")
            user_input = detect_speech_and_transcribe(stream=stream, RATE = RATE, CHUNK = CHUNK, speech_client = speech_client)
            print(f"Mini Pupper has stopped listening.\n")
            print(f"User input: {user_input}\n")

#------------------------------------------------- Chat with Gemini Pro -------------------------------------------------

        # Fetch response from Gemini Pro
        start_time = time.time()
        result = conversation.invoke(input=user_input + "In short. No asterisks, hashtags, or emojis.")['response']
        end_time = time.time()
        print(f"Gemini response time: {end_time - start_time:.2f} seconds\n")

        print(f"Mini Pupper: {result}\n")

        # Set the text to be synthesized
        text_to_speak = result

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
        stream = open_stream(RATE = RATE, CHUNK = CHUNK, p = p)
    
    # 关闭音频流
    if stream.is_active():
        stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
    main()   
