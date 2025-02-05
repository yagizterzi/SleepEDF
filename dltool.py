import mne
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import shap

# Path to the EEG file (example from Sleep-EDF Expanded 2018)
edf_file = r"C:\Users\yagiz\OneDrive\Masaüstü\Uygulamalar\kodlar\SleepEdf\dataset\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf"

# Load EEG data
raw = mne.io.read_raw_edf(edf_file, preload=True)

print(raw.info['ch_names'])
eeg_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']  # Frontal-central EEG recordings
raw.pick_channels(eeg_channels)

hypnogram_dir = r"C:\Users\yagiz\OneDrive\Masaüstü\Uygulamalar\kodlar\SleepEdf\dataset\sleep-edf-database-expanded-1.0.0\sleep-cassette"

# Load and append all hypnogram files
all_hypnogram_dfs = []
for file in os.listdir(hypnogram_dir):
    if file.endswith("-Hypnogram.edf"):
        hypnogram_file = os.path.join(hypnogram_dir, file)
        annotations = mne.read_annotations(hypnogram_file)
        raw.set_annotations(annotations)

        hypno_events = []
        for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
            hypno_events.append([onset, duration, description])

        # Save hypnogram as DataFrame
        hypnogram_df = pd.DataFrame(hypno_events, columns=["Onset", "Duration", "Stage"])
        hypnogram_df["Stage"] = hypnogram_df["Stage"].astype(str)  # Convert stages to string

        # Remap stage values for clarity
        stage_mapping = {
            "Sleep stage W": "W",
            "Sleep stage 1": "N1",
            "Sleep stage 2": "N2",
            "Sleep stage 3": "N3",
            "Sleep stage 4": "N3",
            "Sleep stage R": "REM"
        }
        hypnogram_df["Stage"] = hypnogram_df["Stage"].map(stage_mapping)
        all_hypnogram_dfs.append(hypnogram_df)

# Select a random hypnogram
random_hypnogram_df = random.choice(all_hypnogram_dfs)
print(random_hypnogram_df.head())

# Apply bandpass filter (0.3 - 35 Hz)
raw.filter(l_freq=0.3, h_freq=35)

# Check the sampling frequency
sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")

# Check the Nyquist frequency
nyquist_freq = sfreq / 2
print(f"Nyquist frequency: {nyquist_freq} Hz")

# Apply notch filter (using 49 Hz) if applicable
if nyquist_freq > 49:
    raw.notch_filter(freqs=[49])

# Create epochs
epochs = mne.make_fixed_length_epochs(raw, duration=30.0, overlap=0.0)

features_list = []
for idx, hypnogram_df in enumerate(all_hypnogram_dfs):
    TST = hypnogram_df[hypnogram_df["Stage"] != "W"]["Duration"].sum() / 60  # Total Sleep Time in minutes
    print(f"Hypnogram {idx} - Total Sleep Time (TST): {TST:.2f} minutes")
    
    N3_total = hypnogram_df[hypnogram_df["Stage"] == "N3"]["Duration"].sum()
    N3_percentage = (N3_total / (TST * 60)) * 100 if TST > 0 else 0
    print(f"Hypnogram {idx} - N3 Deep Sleep Percentage: {N3_percentage:.2f}%")
    
    REM_total = hypnogram_df[hypnogram_df["Stage"] == "REM"]["Duration"].sum()
    REM_percentage = (REM_total / (TST * 60)) * 100 if TST > 0 else 0
    print(f"Hypnogram {idx} - REM Percentage: {REM_percentage:.2f}%")
    
    wake_epochs = len(hypnogram_df[hypnogram_df["Stage"] == "W"])
    print(f"Hypnogram {idx} - Number of Awakenings: {wake_epochs}")
    
    non_wake = hypnogram_df[hypnogram_df["Stage"] != "W"]
    SOL = non_wake["Onset"].min() / 60 if not non_wake.empty else 0  # Sleep Onset Latency (minutes)
    print(f"Hypnogram {idx} - Sleep Onset Latency (SOL): {SOL:.2f} minutes")
    
    time_in_bed = hypnogram_df["Duration"].sum() / 60  # Total time in bed in minutes
    sleep_efficiency = (TST / time_in_bed) * 100 if time_in_bed > 0 else 0
    print(f"Hypnogram {idx} - Sleep Efficiency: {sleep_efficiency:.2f}%")
    
    features_list.append({
        "Hypnogram_ID": idx,
        "TST": TST,
        "N3_percentage": N3_percentage,
        "REM_percentage": REM_percentage,
        "Awakenings": wake_epochs,
        "SOL": SOL,
        "Sleep_Efficiency": sleep_efficiency,
    })

df_features = pd.DataFrame(features_list)
print(df_features)

# Build the model
model = Sequential()

# CNN layer - spatial feature extraction
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(df_features.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# LSTM layer - capture temporal dependencies
model.add(LSTM(64, return_sequences=False))  # Only final output is used

# Fully connected (Dense) layer and output layer
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Single output (sleep score)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Display the model summary
model.summary()

# Format features accordingly
X = np.expand_dims(df_features.values, axis=-1)  # Reshape data to [n_samples, n_features, 1]

# Define model target labels. Here, randomly generated values are used; replace with actual sleep scores.
y = np.random.randn(df_features.shape[0])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Visualize the training process
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.legend()
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.show()

# Predict on the test set
y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.plot(y_test, label='Actual Scores', marker='o')
plt.plot(y_pred, label='Predicted Scores', marker='x')
plt.xlabel("Sample")
plt.ylabel("Sleep Score")
plt.title("Actual vs Predicted Sleep Scores")
plt.legend()
plt.show()

