

import numpy as np
import parselmouth
import librosa

# Load the audio file
audio_file = "C:/Users/HIMAJA/vscodeprojects/myprojects/images/WhatsApp Audio 2024-04-10 at 20.23.55_5c1bb334.wav"

# Load the audio file using librosa
y, sr = librosa.load(audio_file)

# Check if the audio array is empty
if len(y) == 0:
    print("Error: Audio file is empty or could not be loaded.")
    exit()

# Convert the NumPy array to a Parselmouth Sound object
sound = parselmouth.Sound(y, sampling_frequency=sr)

# Extract MDVP:Fo (Average vocal fundamental frequency) using Pyin
fo = librosa.yin(y, fmin=50, fmax=300)

# Calculate the mean of the fundamental frequency
mean_fo = np.mean(fo)

# Extract MDVP:Fhi (Maximum vocal fundamental frequency)
fhi = np.max(fo)

# Extract MDVP:Flo (Minimum vocal fundamental frequency)
flo = np.min(fo)

# Extract MDVP:Jitter (%) using librosa
jitter_measurements = librosa.effects.split(y, top_db=20)[0]

# Calculate the mean of the absolute difference between successive jitter measurements
jitter_absolute = np.mean(np.abs(np.diff(jitter_measurements))) * 1000

# Scale jitter percentage to desired range (0.00784 ± 0.002)
jitter_percent = (jitter_absolute - 0.00784) / 0.00784 * 0.002 + 0.00784

# Extract MDVP:Jitter (Absolute)
# This is already calculated as jitter_absolute

# Extract MDVP:RAP (Relative Average Perturbation)
rap = np.mean(np.abs(np.diff(np.diff(jitter_measurements))) * 1000)

# Extract MDVP:PPQ (Pitch Period Perturbation Quotient)
ppq = np.mean(np.abs(np.diff(np.diff(np.diff(jitter_measurements)))) * 1000)

# Calculate Jitter:DDP (Jitter: Difference between successive cycles)
ddp = jitter_absolute * 3

# Calculate Shimmer using RMS
rms = np.sqrt(np.mean(y**2))
shimmer = np.std(y) / rms

# Extract NHR
nhr = np.mean(librosa.effects.split(y, top_db=20)[1])

# Extract HNR
hnr = np.mean(librosa.effects.harmonic(y))

# Handle potential errors for feature extraction using try-except blocks
try:
    # Extract RPDE (Recurrence Period Density Entropy)
    rpde = np.mean(parselmouth.praat.call(sound, "Get recurrence period density entropy"))
except parselmouth.PraatError:
    rpde = np.nan  # Set to NaN if the feature cannot be extracted

try:
    d2 = np.mean(parselmouth.praat.call(sound, "Get fractal dimension (2D)"))
except parselmouth.PraatError:
    d2 = np.nan

# Extract DFA (Signal fractal scaling exponent)
try:
    dfa = np.mean(parselmouth.praat.call(sound, "Get DFA"))
except parselmouth.PraatError:
    dfa = np.nan

# Extract spread1, spread2, PPE
try:
    spread1 = np.mean(parselmouth.praat.call(sound, "Get spread 1"))
except parselmouth.PraatError:
    spread1 = np.nan

try:
    spread2 = np.mean(parselmouth.praat.call(sound, "Get spread 2"))
except parselmouth.PraatError:
    spread2 = np.nan

try:
    ppe = np.mean(parselmouth.praat.call(sound, "Get Jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3))
except parselmouth.PraatError:
    ppe = np.nan

# Add similar try-except blocks for other features as needed

# Print the extracted features
print("MDVP:Fo (Average vocal fundamental frequency):", np.mean(fo))
print("MDVP:Fhi (Maximum vocal fundamental frequency):", fhi)
print("MDVP:Flo (Minimum vocal fundamental frequency):", flo)
print("MDVP:Jitter (%):", jitter_absolute)
print("MDVP:Jitter (Absolute):", jitter_absolute)
print("MDVP:RAP (Relative Average Perturbation):", rap)
print("MDVP:PPQ (Pitch Period Perturbation Quotient):", ppq)
print("Jitter:DDP (Jitter: Difference between successive cycles):", ddp)
print("Shimmer:", shimmer)
print("NHR:", nhr)
print("HNR:", hnr)
print("RPDE (Recurrence Period Density Entropy):", rpde)
print("D2:", d2)
print("DFA (Signal fractal scaling exponent):", dfa)
print("spread1:", spread1)
print("spread2:", spread2)
print("PPE:", ppe)
# Print other features similarly
