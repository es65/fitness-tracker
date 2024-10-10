import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate or load the time series data (example: a sine wave with some noise)
# Let's simulate a simple signal with two frequencies (5Hz and 20Hz)
sampling_rate = 1000  # Samples per second
t = np.linspace(0, 1, sampling_rate, endpoint=False)  # Time vector (1 second)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)  # Combined signal

# Step 2: Perform Fourier Transform using numpy's FFT
fft_result = np.fft.fft(signal)  # FFT result
frequencies = np.fft.fftfreq(len(fft_result), 1 / sampling_rate)  # Frequency bins

# Step 3: Compute magnitude of FFT (optional: only take the positive frequencies)
magnitude = np.abs(fft_result)

# Step 4: Plot the time series and the Fourier transform
plt.figure(figsize=(12, 6))

# Plot the original time series signal
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Original Time Series Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Plot the FFT (only positive frequencies)
plt.subplot(2, 1, 2)
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])  # Only plot positive frequencies
plt.title("Frequency Domain (Fourier Transform)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()
