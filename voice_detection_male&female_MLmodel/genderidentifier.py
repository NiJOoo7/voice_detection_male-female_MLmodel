
import pyaudio
import wave
import librosa
import numpy as np
from tqdm import tqdm
from python_speech_features import mfcc
import pickle

def extract_features(audio_data):
    
    audio_waves = audio_data[:,0]
    samplerate = audio_data[:,1][0]
    if samplerate > 16000:
        samplerate = 16000
    
    features = []
    for audio_wave in tqdm(audio_waves):
        features.append(mfcc(audio_wave, samplerate=samplerate, numcep=26))
        
    features = np.array(features, dtype="object")
    return features

def concatenate_features(audio_features):
    concatenated = audio_features[0]
    for audio_feature in tqdm(audio_features):
        concatenated = np.vstack((concatenated, audio_feature))
        
    return concatenated

model = pickle.load(open("finalmodel.sav", "rb"))

framesperbuffer = 3200
format = pyaudio.paInt16
channels = 1
framerate = 16000

p = pyaudio.PyAudio()
stream = p.open(
    format = format,
    channels = channels,
    rate = framerate,
    input = True,
    frames_per_buffer = framesperbuffer
)

input("Press Enter to continue...")
print("Start recording...")
seconds = 5
frames = []
for i in range(0, int(framerate/framesperbuffer*seconds)):
    data = stream.read(framesperbuffer)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

obj = wave.open("input.wav", "wb")
obj.setnchannels(channels)
obj.setsampwidth(p.get_sample_size(format))
obj.setframerate(framerate)
obj.writeframes(b"".join(frames))
obj.close()

input = []
input.append(librosa.load("D:/5th semester/OSS Project/finalproject/input2.wav"))
input = np.array(input, dtype="object")
input_features = extract_features(input)
input_concatenated = concatenate_features(input_features)

svm_predictions = model.predict(input_concatenated[:1])

if svm_predictions[0] == 0:
    print('Male')
else:
    print('Female')