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

# EEG dosyasının yolu
edf_file = "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf"  # Sleep-EDF Expanded 2018'den bir örnek dosya

# EEG verisini yükle
raw = mne.io.read_raw_edf(edf_file, preload=True)

print(raw.info['ch_names'])
eeg_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']  # Ön-orta bölge EEG kaydı
raw.pick_channels(eeg_channels)

hypnogram_dir = "sleep-edf-database-expanded-1.0.0\sleep-cassette"

# Tüm hypnogram dosyalarını yükle ve ekle
all_hypnogram_dfs = []
for file in os.listdir(hypnogram_dir):
    if file.endswith("-Hypnogram.edf"):
        hypnogram_file = os.path.join(hypnogram_dir, file)
        annotations = mne.read_annotations(hypnogram_file)
        raw.set_annotations(annotations)

        hypno_events = []
        for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
            hypno_events.append([onset, duration, description])

        # Hypnogram'ı DataFrame olarak kaydet
        hypnogram_df = pd.DataFrame(hypno_events, columns=["Onset", "Duration", "Stage"])
        hypnogram_df["Stage"] = hypnogram_df["Stage"].astype(str)  # Stage'leri string formatına çevir

        # Uyku evrelerini daha anlaşılır hale getirmek için Stage sütunundaki değerleri düzenleyelim
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

# Rastgele bir hypnogram seç
random_hypnogram_df = random.choice(all_hypnogram_dfs)
print(random_hypnogram_df.head())

# Bant geçiren filtre (0.3 - 35 Hz)
raw.filter(l_freq=0.3, h_freq=35)

# Örnekleme frekansını kontrol et
sfreq = raw.info['sfreq']
print(f"Örnekleme frekansı: {sfreq} Hz")

# Nyquist frekansını kontrol et
nyquist_freq = sfreq / 2
print(f"Nyquist frekansı: {nyquist_freq} Hz")

# Notch filtresi uygula (49 Hz frekansını kullan)
if nyquist_freq > 49:
    raw.notch_filter(freqs=[49])

# Epoch'ları oluştur
epochs = mne.make_fixed_length_epochs(raw, duration=30.0, overlap=0.0)

features_list = []
for idx, hypnogram_df in enumerate(all_hypnogram_dfs):
    TST = hypnogram_df[hypnogram_df["Stage"] != "W"]["Duration"].sum() / 60  # Convert to minutes
    print(f"Hypnogram {idx} - Toplam Uyku Süresi (TST): {TST:.2f} dakika")
    
    N3_total = hypnogram_df[hypnogram_df["Stage"] == "N3"]["Duration"].sum()
    N3_percentage = (N3_total / (TST * 60)) * 100 if TST > 0 else 0
    print(f"Hypnogram {idx} - N3 Derin Uyku Yüzdesi: {N3_percentage:.2f}%")
    
    REM_total = hypnogram_df[hypnogram_df["Stage"] == "REM"]["Duration"].sum()
    REM_percentage = (REM_total / (TST * 60)) * 100 if TST > 0 else 0
    print(f"Hypnogram {idx} - REM Yüzdesi: {REM_percentage:.2f}%")
    
    wake_epochs = len(hypnogram_df[hypnogram_df["Stage"] == "W"])
    print(f"Hypnogram {idx} - Gece Uyanma Sayısı: {wake_epochs}")
    
    non_wake = hypnogram_df[hypnogram_df["Stage"] != "W"]
    SOL = non_wake["Onset"].min() / 60 if not non_wake.empty else 0  # Convert to minutes
    print(f"Hypnogram {idx} - Uyku Başlangıç Gecikmesi (SOL): {SOL:.2f} dakika")
    
    time_in_bed = hypnogram_df["Duration"].sum() / 60  # Total time in bed in minutes
    sleep_efficiency = (TST / time_in_bed) * 100 if time_in_bed > 0 else 0
    print(f"Hypnogram {idx} - Uyku Verimliliği: {sleep_efficiency:.2f}%")
    
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


# Modeli kurma
model = Sequential()

# CNN katmanı - uzamsal özellik çıkarımı
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(df_features.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# LSTM katmanı - zaman bağımlılıklarını yakalama
model.add(LSTM(64, return_sequences=False))  # return_sequences=False, sadece son çıktıyı alacak

# Tam bağlantılı katman (Dense) ve çıkış katmanı
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Tek bir çıkış (uyku skoru)

# Modelin derlenmesi
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Modeli özetleme
model.summary()

# Özelliklerin uygun şekilde formatlanması
X = np.expand_dims(df_features.values, axis=-1)  # Veriyi [n_samples, n_features, 1] şeklinde yapılandır

# Modelin hedef etiketini belirleyin. Bu örnekte, rastgele oluşturuldu, gerçek uyku skorları ile değiştirin.
y = np.random.randn(df_features.shape[0])  # Gerçek uyku skoru burada kullanılacak

# Eğitim ve test veri setleri olarak bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Eğitim sürecinin görselleştirilmesi
plt.plot(history.history['loss'], label='Eğitim kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama kaybı')
plt.legend()
plt.title('Model Kaybı (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['mae'], label='Eğitim MAE')
plt.plot(history.history['val_mae'], label='Doğrulama MAE')
plt.legend()
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.show()

# Test seti üzerinden tahmin yapma
y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.plot(y_test, label='Gerçek Skorlar', marker='o')
plt.plot(y_pred, label='Tahmin Edilen Skorlar', marker='x')
plt.xlabel("Örnek")
plt.ylabel("Uyku Skoru")
plt.title("Gerçek ve Tahmin Edilen Uyku Skorları")
plt.legend()
plt.show()

