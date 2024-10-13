import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Sample time series data creation (replace this with your actual DataFrame and column)
# For example, a signal with a mixture of low and high frequency components
sampling_rate = 100  # Samples per second
t = np.linspace(0, 1, sampling_rate, endpoint=False)
# Create a signal with two frequencies: 5 Hz (low) and 50 Hz (high)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

# Create a DataFrame
df = pd.DataFrame({'time': t, 'signal': signal})

# Plot the original signal
plt.figure(figsize=(10, 4))
plt.plot(df['time'], df['signal'], label='Original Signal')
plt.title('Original Time Series Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Step 1: Define the Butterworth lowpass filter
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a Butterworth lowpass filter to the given data.

    Args:
        data (array-like): The input signal to filter.
        cutoff (float): The cutoff frequency for the lowpass filter.
        fs (float): The sampling rate of the signal (samples per second).
        order (int): The order of the filter (default is 4).

    Returns:
        array-like: The filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency is half of the sampling rate
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Step 2: Apply the lowpass filter to the signal
cutoff_frequency = 10  # Set the desired cutoff frequency (in Hz)
filtered_signal = butter_lowpass_filter(df['signal'], cutoff=cutoff_frequency, fs=sampling_rate)

# Step 3: Add the filtered signal back to the DataFrame
df['filtered_signal'] = filtered_signal

# Step 4: Plot the filtered signal
plt.figure(figsize=(10, 4))
plt.plot(df['time'], df['signal'], label='Original Signal', alpha=0.5)
plt.plot(df['time'], df['filtered_signal'], label='Filtered Signal (Lowpass)', color='red')
plt.title('Filtered Time Series Signal (Butterworth Lowpass)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
