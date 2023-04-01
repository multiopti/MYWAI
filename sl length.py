# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from numpy import linalg as LA

# Set the sampling rate and duration
Fs = 20  # Sampling rate (Hz)
duration = 10  # Duration of the signal (seconds)

# Create a time vector
t = np.arange(0, duration, 1/Fs)

# Create the sine wave signals with frequencies of 1Hz and 10Hz
freq1 = 0.1  # Frequency of the first sine wave (Hz)
freq2 = 1  # Frequency of the second sine wave (Hz)
signal1 = np.sin(2 * np.pi * freq1 * t)
label1 = np.ones(len(signal1))
signal2 = np.sin(2 * np.pi * freq2 * t)
label2 = np.zeros(len(signal1))

# Create the alternating signal
num_periods = 2000
period = np.concatenate([signal1, signal2])
plabel = np.concatenate([label1, label2])
alternating_signal = np.tile(period, num_periods)
labels = np.tile(plabel, num_periods)

# Add Gaussian noise to the signal
mean = 0
stddev = 0.1
noise = np.random.normal(mean, stddev, alternating_signal.shape)
noisy_signal = alternating_signal + noise

ta = np.arange(len(alternating_signal))
# Plot the signal
plt.plot(ta[0:1000], noisy_signal[0:1000])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

    
# Define the number of columns and the number of positions to shift for each column
num_cols = 200

# Initialize the output array with zeros
output_array = np.zeros((len(noisy_signal), num_cols))

# Fill the first column of the output array with the noisy signal
output_array[:, 0] = noisy_signal

# Fill the remaining columns by shifting the noisy signal
# Fill the remaining columns by shifting the noisy signal
for i in range(1, num_cols):
    output_array[:, i] = np.roll(noisy_signal, i)
    
data = output_array
    
# Create a training dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

# Create an MLPClassifier with 50 neurons in the hidden layer and a linear activation function
clf = MLPRegressor(hidden_layer_sizes=(10),solver='sgd',max_iter=20000,verbose=True, tol=1e-4,activation='relu',  n_iter_no_change=100, random_state=12)

# Train the classifier on the training dataset
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing dataset
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)

