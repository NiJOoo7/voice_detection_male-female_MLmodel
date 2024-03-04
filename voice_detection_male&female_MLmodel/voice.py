import os
import pyaudio
import wave
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from pydub import AudioSegment
from python_speech_features import mfcc
from time import time
import pickle

df = pd.read_csv('D:/5th semester/OSS Project/dataset/cv-valid-train.csv')


df_male = df[df['gender']=='male']
df_female = df[df['gender']=='female']

# print(df_male.shape)		

# print(df_female.shape)		

df_male = df_male[:300]
df_female = df_female[:300]

TRAIN_PATH = 'D:/5th semester/OSS Project/dataset/cv-valid-train/'

def convert_to_wav(df, m_f, path=TRAIN_PATH):
    
    for file in tqdm(df['filename']):
        sound = AudioSegment.from_mp3(path+file)
        #print('male-'+file.split('/')[-1].split('.')[0]+'.wav')
        
		# Create new wav files based on existing mp3 files
        if m_f == 'male':
             sound.export('D:/5th semester/OSS Project/finalproject/WAVS/male-'+file.split('/')[-1].split('.')[0]+'.wav', format='wav')
        elif m_f == 'female':
             sound.export('D:/5th semester/OSS Project/finalproject/WAVS/female-'+file.split('/')[-1].split('.')[0]+'.wav', format='wav')
        
    return

convert_to_wav(df_male, m_f='male')
convert_to_wav(df_female, m_f='female')


def load_audio(audio_files):
    male_voices = []
    female_voices = []

    for file in tqdm(audio_files):
        if file.split('-')[0] == 'male':
            male_voices.append(librosa.load('D:/5th semester/OSS Project/finalproject/WAVS/' + file))
        elif file.split('-')[0] == 'female':
            female_voices.append(librosa.load('D:/5th semester/OSS Project/finalproject/WAVS/' + file))
    
    male_voices = np.array(male_voices, dtype="object")
    female_voices = np.array(female_voices, dtype="object")
    
    return male_voices, female_voices

male_voices, female_voices = load_audio(os.listdir("D:/5th semester/OSS Project/finalproject/WAVS"))


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

male_features = extract_features(male_voices)
female_features = extract_features(female_voices)


def concatenate_features(audio_features):
    concatenated = audio_features[0]
    for audio_feature in tqdm(audio_features):
        concatenated = np.vstack((concatenated, audio_feature))
        
    return concatenated

male_concatenated = concatenate_features(male_features)
female_concatenated = concatenate_features(female_features)

# print(male_concatenated.shape)

# print(female_concatenated.shape)


X = np.vstack((male_concatenated, female_concatenated))

y = np.append([0] * len(male_concatenated), [1] * len(female_concatenated))

# print(X.shape)

# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)


values = []
train_acc = []
test_acc = []

# Initialize SVM model rbf
clf1 = svm.SVC(kernel='rbf')

# Train the model
start = time()
clf1.fit(X_train[:5000], y_train[:5000])
values.append(time() - start)
# print(time()-start)

# Compute the accuracy score towards train data
start = time()
train_acc.append(clf1.score(X_train[:5000], y_train[:5000]))
# print(clf.score(X_train[:5000], y_train[:5000]))

# print(time()-start)

# Compute the accuracy score towards test data
start = time()
test_acc.append(clf1.score(X_test[:1000], y_test[:1000]))
# print(clf.score(X_test[:1000], y_test[:1000]))

# print(time()-start)


# Predict the first 10000 test data
svm_predictions = clf1.predict(X_test[:1000])

# Create the confusion matrix values
cm = confusion_matrix(y_test[:1000], svm_predictions)

# Create the confusion matrix display
plt.figure(figsize=(8,8))
plt.title('Confusion matrix on test data (SVM-RBF)')
sns.heatmap(cm, annot=True, fmt='d', 
            cmap=plt.cm.Blues, cbar=False, annot_kws={'size':14})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

'''filename = "finalmodel.sav"
pickle.dump(clf1, open(filename, "wb"))'''


# Initialize SVM model poly
clf = svm.SVC(kernel="poly")      

# Train the model
start = time()
clf.fit(X_train[:5000], y_train[:5000])
values.append(time() - start)
# print(time()-start)

# Compute the accuracy score towards train data
start = time()
train_acc.append(clf.score(X_train[:5000], y_train[:5000]))
# print(clf.score(X_train[:5000], y_train[:5000]))

# print(time()-start)

# Compute the accuracy score towards test data
start = time()
test_acc.append(clf.score(X_test[:1000], y_test[:1000]))
# print(clf.score(X_test[:1000], y_test[:1000]))

# print(time()-start)


# Predict the first 10000 test data
svm_predictions = clf.predict(X_test[:1000])

# Create the confusion matrix values
cm = confusion_matrix(y_test[:1000], svm_predictions)

# Create the confusion matrix display
plt.figure(figsize=(8,8))
plt.title('Confusion matrix on test data (SVM-Poly)')
sns.heatmap(cm, annot=True, fmt='d', 
            cmap=plt.cm.Blues, cbar=False, annot_kws={'size':14})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Initialize SVM model sigmoid
clf = svm.SVC(kernel="sigmoid")      

# Train the model
start = time()
clf.fit(X_train[:5000], y_train[:5000])
values.append(time() - start)
# print(time()-start)

# Compute the accuracy score towards train data
start = time()
train_acc.append(clf.score(X_train[:5000], y_train[:5000]))
# print(clf.score(X_train[:5000], y_train[:5000]))

# print(time()-start)

# Compute the accuracy score towards test data
start = time()
test_acc.append(clf.score(X_test[:1000], y_test[:1000]))
# print(clf.score(X_test[:1000], y_test[:1000]))

# print(time()-start)


# Predict the first 10000 test data
svm_predictions = clf.predict(X_test[:1000])

# Create the confusion matrix values
cm = confusion_matrix(y_test[:1000], svm_predictions)

# Create the confusion matrix display
plt.figure(figsize=(8,8))
plt.title('Confusion matrix on test data (SVM-Sigmoid)')
sns.heatmap(cm, annot=True, fmt='d', 
            cmap=plt.cm.Blues, cbar=False, annot_kws={'size':14})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Performance comparison between different algorithms
index = ['SVM-RBF', 'SVM-Poly', 'SVM-Sigmoid']

plt.figure(figsize=(12,3))
plt.title('Training duration (lower is better)')
plt.xlabel('Seconds')
plt.ylabel('Model')
plt.barh(index, values, zorder=2)
plt.grid(zorder=0)

for i, value in enumerate(values):
    plt.text(value+20, i, str(value)+' secs', fontsize=12, color='black',
             horizontalalignment='center', verticalalignment='center')

plt.show()

barWidth = 0.25
    
index = ['SVM-RBF', 'SVM-Poly', 'SVM-Sigmoid']

baseline = np.arange(len(train_acc))
r1 = [x + 0.125 for x in baseline]
r2 = [x + 0.25 for x in r1]

plt.figure(figsize=(16,9))
plt.title('Model performance (higher is better)')
plt.bar(r1, train_acc, width=barWidth, label='Train', zorder=2)
plt.bar(r2, test_acc, width=barWidth, label='Test', zorder=2)
plt.grid(zorder=0)

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks([r + barWidth for r in range(len(train_acc))], index)

for i, value in enumerate(train_acc):
    plt.text(i+0.125, value-5, str(value), fontsize=12, color='white',
             horizontalalignment='center', verticalalignment='center')
    
for i, value in enumerate(test_acc):
    plt.text(i+0.375, value-5, str(value), fontsize=12, color='white',
             horizontalalignment='center', verticalalignment='center')
    
plt.legend()
plt.show()

'''framesperbuffer = 3200
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
input.append(librosa.load("D:/5th semester/OSS Project/finalproject/input.wav"))
input = np.array(input, dtype="object")
input_features = extract_features(input)
input_concatenated = concatenate_features(input_features)

svm_predictions = clf1.predict(input_concatenated[:1])

if svm_predictions[0] == 0:
    print('Male')
else:
    print('Female')'''