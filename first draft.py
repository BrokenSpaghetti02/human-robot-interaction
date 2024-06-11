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

while True:
#------------------------------------------------- Speech to Text -------------------------------------------------
    
    # Audio record parameters
    fs = 48000  # 48KHz,Audio sampling rate
    duration = 5  # Recording duration in seconds

    # Record audio
    print("Mini Pupper 2 is listening...\n")
    start_time = time.time()
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait for record to finish
    end_time = time.time()
    print(f"Mini Pupper has stopped listening. Time taken: {end_time - start_time:.2f} seconds\n")

    # Increase the volume [manually]
    recording *= 80

    # Save the recording
    file_path = '/home/ubuntu/mini-pupper-2-audio_test.wav'
    sf.write(file_path, recording, fs, subtype = 'PCM_16', format = 'WAV')

    # Perform STT transcription
    start_time = time.time()
    user_input = transcribe_file_with_enhanced_model(file_path)
    end_time = time.time()
    print(f"Speech to Text processing time: {end_time - start_time:.2f} seconds\n")
    print(f"User input: {user_input}\n")

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