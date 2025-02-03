
# EEG Sleep Stage Prediction using Deep Learning

This project aims to predict a person's sleep stages using EEG (Electroencephalography) data, leveraging machine learning and deep learning techniques. The sleep stages represent different brainwave activities during sleep, and this project processes EEG signals to classify these stages.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Data Description](#data-description)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Incorrect Predictions Analysis](#incorrect-predictions-analysis)
- [Testing with New Data](#testing-with-new-data)
- [Conclusion](#conclusion)

## Project Overview

The goal of this project is to classify sleep stages from EEG data using a Convolutional Neural Network (CNN). The model is trained on labeled EEG data and learns to predict the sleep stages based on the brainwave patterns it detects. The project includes data preprocessing, model training, evaluation, and analysis of predictions.

## Getting Started

To run this project, you will need Python 3.x and the following libraries:

- `numpy`
- `mne`
- `tensorflow`
- `scikit-learn`
- `matplotlib`
  
You can install the required libraries using `pip`:

```bash
pip install numpy mne tensorflow scikit-learn matplotlib
```

Once you have the dependencies installed, you can start by loading the EEG and hypnogram data files into the project.

## Data Description

This project uses EEG data in `.edf` (European Data Format) files, which are commonly used for storing brainwave measurements. The data contains signals from specific EEG channels and labels indicating the sleep stage.

- **EEG data**: It contains the brain activity for specific channels recorded during sleep.
- **Hypnogram data**: This provides annotations for the sleep stages at different time intervals.

Sleep stages are labeled as follows:
- **W (Wake)**: Wakefulness
- **1**: Light Sleep
- **2**: Deep Sleep
- **3/4**: Very Deep Sleep
- **R (REM)**: Rapid Eye Movement sleep (associated with dreaming)

## Model Architecture

The model is built using a **Convolutional Neural Network (CNN)**, which is effective for time-series data like EEG signals. The architecture includes the following layers:

1. **Conv1D**: Detects patterns in the EEG signals corresponding to different sleep stages.
2. **MaxPooling1D**: Reduces dimensionality to speed up training.
3. **Flatten**: Converts the 2D output of the convolution layers into a 1D vector.
4. **Dense**: Fully connected layer that learns high-level features.
5. **Dropout**: Prevents overfitting by randomly setting a fraction of input units to 0 at each update.
6. **Softmax**: Outputs a probability distribution for the sleep stages.

The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.

## Training the Model

After preprocessing the EEG data, the model is trained using the training set. The training process involves adjusting the model's weights to minimize the error in predicting sleep stages.

```python
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

## Evaluating the Model

Once trained, the model is evaluated on a separate test set to determine its accuracy and loss:

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
```

This provides an indication of how well the model generalizes to new, unseen data.

## Incorrect Predictions Analysis

The model's performance is further analyzed by examining its incorrect predictions. This helps in understanding where the model made mistakes and identifying any potential improvements.

```python
incorrect_predictions = []
for i in range(len(X_test)):
    sample_input = np.expand_dims(X_test[i], axis=0)
    predicted_label = np.argmax(model.predict(sample_input))
    if predicted_label != y_test[i]:
        incorrect_predictions.append((i, y_test[i], predicted_label))
```

## Testing with New Data

To test the model with new EEG data that it hasn't seen before, you can use the following function:

```python
def test_new_eeg(new_eeg_path):
    raw_new = mne.io.read_raw_edf(new_eeg_path, preload=True)
    raw_new.pick_channels(['EEG Fpz-Cz', 'EEG Pz-Oz'])
    new_data = raw_new.get_data()
    new_data = scaler.transform(new_data.T).T  
    new_data = np.expand_dims(new_data.T, axis=-1)
    prediction = np.argmax(model.predict(new_data), axis=1)
    print(f"Predicted Sleep Stages: {prediction}")
```

This function takes a new EEG file as input, processes it, and predicts the sleep stage.

## Conclusion

This project successfully predicts sleep stages using EEG data and deep learning techniques. The model leverages a Convolutional Neural Network (CNN) to identify patterns in the brainwave activity associated with different sleep stages. This work can be applied in healthcare settings to monitor and analyze sleep patterns, and potentially assist with diagnosing sleep disorders.

---

Feel free to contribute to this project or use it as a base for further research on EEG-based sleep stage classification.
```

This README provides a detailed, step-by-step explanation of the project,
