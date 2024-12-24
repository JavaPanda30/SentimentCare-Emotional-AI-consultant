import pandas as pd
import numpy as np
import tensorflow as tf  
from tensorflow.keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import pickle

data = pd.read_csv("train_tweet.csv", sep=';')
data.columns = ["Text", "Emotions"]

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
one_hot_labels = tf.keras.utils.to_categorical(labels)

xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences,one_hot_labels, test_size=0.45)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length))
model.add(GlobalAveragePooling1D())  
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=len(one_hot_labels[0]), activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(xtrain, ytrain, epochs=15, batch_size=32, validation_data=(xtest, ytest))
model_architecture = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_architecture)

model.save_weights("model_weights.weights.h5")  

with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)
    