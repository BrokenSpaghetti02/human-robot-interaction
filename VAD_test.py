import os
import pyaudio
from google.cloud import speech_v1p1beta1 as speech

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/allen/Downloads/MangDang/本地语音交互/super_key.json'

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks

client = speech.SpeechClient()

# configuration
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code="en-US" 
)

streaming_config = speech.StreamingRecognitionConfig(
    config=config,
    interim_results=True
)

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

def audio_generator():
    while True:
        data = stream.read(CHUNK)
        yield data

print("Listening...")  # Start listening
requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator())
responses = client.streaming_recognize(requests=requests, config=streaming_config)

# deal with streaming
utterance_buffer = []

for response in responses:
    for result in response.results:
        if result.is_final:
            # adding the final result
            if result.alternatives:
                utterance_buffer.append(result.alternatives[0].transcript)
            # end of the voice，deal with sth in utterance_buffer
            utterance = " ".join(utterance_buffer)
            print(f"Recognition result: {utterance}") 
            utterance_buffer = []

# turn off
print("Stopping...") 
stream.stop_stream()
stream.close()
p.terminate()