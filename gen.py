import numpy as np
import tensorflow as tf
from tensorflow import keras

# Sample text
text = "Hello, how are you doing today?"

# Create a character mapping
chars = sorted(set(text))
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}

# Prepare training data
input_text = [text[i:i+3] for i in range(len(text)-3)]
output_text = [text[i+3] for i in range(len(text)-3)]

x = np.array([[[char_to_index[char] for char in seq]] for seq in input_text])
y = np.array([char_to_index[char] for char in output_text])

# Build the RNN model
model = keras.Sequential([
    keras.layers.SimpleRNN(64, input_shape=(3, 1)),
    keras.layers.Dense(len(chars), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(x, y, epochs=1000)

# Generate text
start = "Hel"  # Seed text
generated_text = start

for _ in range(20):
    x_pred = np.array([[[char_to_index[char] for char in start]]])
    prediction = model.predict(x_pred)
    next_index = np.argmax(prediction)
    next_char = index_to_char[next_index]
    generated_text += next_char
    start = start[1:] + next_char

print(generated_text)
