import os
import google.auth
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
import base64
from langchain_google_community import SpeechToTextLoader
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
from pydub import AudioSegment
from google.cloud import speech
import sys
import collections
import wave
import pyaudio
import webrtcvad

#------------------------------------------------- Initialization and Definition -------------------------------------------------

# Get Credentials
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
    model_name = 'gemini-pro',
    convert_system_message_to_human = True,
)

# Initialize Prompt
prompt = ChatPromptTemplate(
    messages = [SystemMessagePromptTemplate.from_template(
        """
        You are a robot puppy who is a helpful AI assistant. Your name is Mini Pupper V2.
        Yor must always answer in short and direct style unless necessary.
        """
    ),
    MessagesPlaceholder(variable_name = "history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

# Set memory for context
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation = ConversationChain(llm=model, prompt=prompt, verbose=False, memory=memory)

# Defining function to transcribe
def transcribe_file_with_enhanced_model(path: str) -> speech.RecognizeResponse:
    """Transcribe the given audio file using an enhanced model."""

    client = speech.SpeechClient()

    # path = 'resources/commercial_mono.wav'
    with open(path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="en-US",
        use_enhanced=True,
        # A model must be specified to use enhanced model.
        model="phone_call",
    )

    response = client.recognize(config=config, audio=audio)
    
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        return alternative.transcript

# Set the default speaker volume to maximum
# Headphone number is 0 without HDMI output
# Headphone number is 1 when HDMI connect the display
os.system("amixer -c 0 sset 'Headphone' 100%")

# Create the TextToSpeechClient instance
client = texttospeech.TextToSpeechClient()

print("Initialization Complete\n")
time.sleep(1)

speak = str(input("Speak to Mini Pupper? (y/n)\n"))

while speak == 'n':
    speak = str(input("Speak to Mini Pupper? (y/n)\n"))

time.sleep(1)

def record_audio(sample_rate, frame_duration_ms, padding_duration_ms, vad):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                   channels=1,
                   rate=sample_rate,
                   input=True,
                   frames_per_buffer=int(frame_duration_ms * sample_rate / 1000))

    print("Recording started. Press Ctrl+C to stop.")
    try:
        while True:
            frame = stream.read(int(frame_duration_ms * sample_rate / 1000))
            is_speech = vad.is_speech(frame, sample_rate)

            sys.stdout.write('1' if is_speech else '0')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    sys.stdout.write('+')
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    sys.stdout.write('-')
                    triggered = False
                    break
    except KeyboardInterrupt:
        sys.stdout.write('-')
        triggered = False

    print("\nRecording stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b''.join(voiced_frames)

#------------------------------------------------- Speech to Text -------------------------------------------------

while True:
    vad = webrtcvad.Vad(1)
    file_path = "/home/ubuntu/recorded_audio.wav"
    print("Mini Pupper 2 is listening...\n")
    start_time = time.time()
    audio = record_audio(sample_rate=48000, frame_duration_ms=30, padding_duration_ms=300, vad=vad)
    end_time = time.time()
    print(f"Mini Pupper has stopped listening. Time taken: {end_time - start_time:.2f} seconds\n")
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(audio)

    # Perform STT transcription
    start_time = time.time()
    user_input = transcribe_file_with_enhanced_model(file_path)
    end_time = time.time()
    print(f"Speech to Text processing time: {end_time - start_time:.2f} seconds\n")
    print(f"User input: {user_input}\n")

    if user_input == None:
        print("No input detected, chat rejected!")
        break

#------------------------------------------------- Chat with Gemini Pro -------------------------------------------------

    # Fetch response from Gemini Pro
    start_time = time.time()
    result = conversation.invoke(input = user_input)['response']
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
    response = client.synthesize_speech(
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