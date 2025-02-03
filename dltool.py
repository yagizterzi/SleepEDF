import numpy as np
import mne
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random


# 1. EEG Verisini Yükle
eeg_path = "SleepEdf\SC4001E0-PSG.edf"
raw_eeg = mne.io.read_raw_edf(eeg_path, preload=True)
raw_eeg.pick(['EEG Fpz-Cz', 'EEG Pz-Oz'])  # İlgili kanalları seç
data = raw_eeg.get_data()
sfreq = raw_eeg.info['sfreq']

# 2. Hypnogramı Yükle ve İşle
hypnogram_path = "SleepEdf\SC4002EC-Hypnogram.edf"
raw_hyp = mne.io.read_raw_edf(hypnogram_path, preload=False)
events, event_id = mne.events_from_annotations(raw_hyp)
print(events)
print(event_id)



# Açıklamaları etiketlere dönüştür
sleep_stage_mapping = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
}
labels = []
for ann in annotations:
    stage = ann['description']
    if stage in sleep_stage_mapping:
        n_epochs = int(ann['duration'] / 30)  # 30 saniyelik epoch sayısı
        labels.extend([sleep_stage_mapping[stage]] * n_epochs)

# 3. EEG ve Etiket Verilerini Eşitle
epoch_duration = 30  # saniye
n_samples_per_epoch = int(sfreq * epoch_duration)
n_epochs = int(data.shape[1] / n_samples_per_epoch)

# EEG verisini yeniden şekillendir (epoch, zaman, kanal)
X = data.T.reshape(n_epochs, n_samples_per_epoch, -1)

# Etiketleri EEG ile aynı boyuta getir
labels = np.array(labels[:n_epochs])  # EEG'ye göre kes
# 4. Veriyi Ölçeklendir
scaler = StandardScaler()
X = np.array([scaler.fit_transform(epoch) for epoch in X])
print(f"X shape: {X.shape}, Labels length: {len(labels)}")
# 5. Veriyi Eğitim ve Test Kümelerine Ayır
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, shuffle=True, random_state=42)


# 6. Model Mimarisini Oluştur
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

# 7. Modeli Derle ve Eğit
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 8. Modeli Değerlendir
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")


# 7. Modeli değerlendirme
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# 8. Modeli test etme (rastgele bir örnek seçerek)
random_idx = random.randint(0, len(X_test) - 1)
sample_input = np.expand_dims(X_test[random_idx], axis=0)
predicted_label = np.argmax(model.predict(sample_input))
true_label = y_test[random_idx]
print(f"Gerçek: {true_label}, Tahmin: {predicted_label}")

# 9. Yanlış tahminleri analiz etme
incorrect_predictions = []
for i in range(len(X_test)):
    sample_input = np.expand_dims(X_test[i], axis=0)
    predicted_label = np.argmax(model.predict(sample_input))
    if predicted_label != y_test[i]:
        incorrect_predictions.append((i, y_test[i], predicted_label))

print(f"Yanlış tahmin edilen örnek sayısı: {len(incorrect_predictions)}")
print("İlk 5 yanlış tahmin:")
for i, true_label, predicted_label in incorrect_predictions[:5]:
    print(f"Örnek {i} -> Gerçek: {true_label}, Tahmin: {predicted_label}")

# 10. Yeni bir EEG sinyali ile test etme
def test_new_eeg(new_eeg_path):
    raw_new = mne.io.read_raw_edf(new_eeg_path, preload=True)
    raw_new.pick_channels(['EEG Fpz-Cz', 'EEG Pz-Oz'])
    new_data = raw_new.get_data()
    new_data = scaler.transform(new_data.T).T  
    new_data = np.expand_dims(new_data.T, axis=-1)
    prediction = np.argmax(model.predict(new_data), axis=1)
    print(f"Tahmin Edilen Uyku Evreleri: {prediction}")
