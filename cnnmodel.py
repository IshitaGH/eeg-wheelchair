import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

input_shape = (250, 8)  #1-second EEG segment, 8 channels -> need at least 1 second per segment

model = Sequential([
    Conv1D(16, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  # 5 classes: Forward, Backward, Left, Right, Stop
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])