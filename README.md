
# EEG & Hypnogram Sleep Quality Scoring with CNN + LSTM

This repository contains code for a deep learning project that predicts sleep quality scores from EEG signals and hypnogram data. The model uses a combined CNN + LSTM architecture to extract spatial and temporal features from sleep recordings and computes key sleep metrics. The predicted sleep quality score is inspired by the Epworth Sleepiness Scale.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation & Analysis](#evaluation--analysis)
- [Future Work](#future-work)
- [License](#license)

## Overview

In this project, we process EEG data from the Sleep-EDF Expanded 2018 dataset and extract features from the corresponding hypnograms. The features include:
- **Total Sleep Time (TST)**
- **N3 Sleep Percentage** (Deep sleep percentage)
- **REM Sleep Percentage**
- **Number and duration of awakenings**
- **Sleep Onset Latency (SOL)**
- **Sleep Efficiency**

We use these features to train a deep learning model based on a CNN + LSTM architecture. The CNN layers extract spatial features from the signal, and the LSTM layers capture the temporal dependencies. The model is then evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Dataset

- **Dataset:** Sleep-EDF Expanded 2018(https://www.physionet.org/content/sleep-edfx/1.0.0/)
- **Files:** The repository expects EEG recordings in EDF format (e.g., `SC4001E0-PSG.edf`) and corresponding hypnogram files (e.g., `*-Hypnogram.edf`).

> **Note:** Ensure you have the dataset in the correct folder structure. The current code uses local paths such as:
> ```
> C:\Users\yagiz\OneDrive\Masaüstü\Uygulamalar\kodlar\SleepEdf\dataset\sleep-edf-database-expanded-1.0.0\sleep-cassette\
> ```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── code
│   └── dltool.py           # Main code for data processing and model training
└── dataset                 # Folder containing the Sleep-EDF Expanded 2018 EDF files
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yagizterzi/SleepEDF]
   cd SleepEDF
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Example `requirements.txt`:**
   ```
   mne
   numpy
   pandas
   scipy
   matplotlib
   seaborn
   tensorflow
   scikit-learn
   shap
   optuna
   ```

## Usage

1. **Data Preparation & Feature Extraction:**

   The script reads the EEG and hypnogram EDF files from the dataset folder. It performs the following:
   - Loads EEG signals and selects the relevant channels (`EEG Fpz-Cz` and `EEG Pz-Oz`).
   - Loads and maps the hypnogram annotations to sleep stages (W, N1, N2, N3, REM).
   - Applies a band-pass filter (0.3-35 Hz) and a notch filter (49 Hz) to clean the EEG data.
   - Segments the data into 30-second epochs.
   - Extracts sleep metrics such as Total Sleep Time (TST), N3 percentage, REM percentage, awakenings, SOL, and sleep efficiency.

2. **Model Training:**

   The CNN + LSTM model is defined and compiled using TensorFlow. The training pipeline:
   - Formats the extracted features as input (reshaped to `[n_samples, n_features, 1]`).
   - Uses a randomly generated target sleep quality score (replace this with your actual labels as needed).
   - Splits the data into training and testing sets.
   - Trains the model for 100 epochs with early stopping using a validation split.
   - Visualizes the loss and MAE during training.
   - Evaluates the model on the test set and plots the predicted versus true sleep scores.

   To run the model training, simply execute:
   ```bash
   python code/dltool.py
   ```

3. **Model Evaluation & Feature Importance Analysis:**

   The repository includes code to:
   - Compare model predictions with actual sleep scores using Pearson correlation.
   - Evaluate sleep stage predictions using metrics like accuracy, precision, recall, and F1-score.
   - Use SHAP to analyze feature importance and understand which sleep metrics most affect the predicted sleep quality.

## Model Architecture

The model architecture is as follows:

- **Conv1D Layer:** Extracts spatial features from the input.
- **MaxPooling1D & Dropout Layers:** Reduce dimensionality and prevent overfitting.
- **LSTM Layer:** Captures temporal dependencies in the sequence data.
- **Dense Layers:** Further processes the features and outputs the final sleep quality score.

The model is compiled with the Adam optimizer and uses mean squared error (MSE) as the loss function.

## Evaluation & Analysis

- **Sleep Score Comparison:**  
  The model's predicted sleep scores are compared with true sleep scores (e.g., based on the Epworth Sleepiness Scale) using Pearson correlation.

- **Sleep Stage Analysis:**  
  The code compares the predicted sleep stages with the actual hypnogram stages using accuracy, precision, recall, and F1-score metrics.

- **Feature Importance with SHAP:**  
  SHAP values are computed to determine the contribution of each feature (TST, N3 percentage, REM percentage, etc.) to the final sleep score prediction.

## Future Work

- **Data Augmentation:** Incorporate more patient data or multiple nights per patient to enhance model robustness.
- **Advanced Hyperparameter Optimization:** Use Bayesian optimization frameworks like Optuna for fine-tuning model parameters.
- **Deployment:** Wrap the model in a Flask API for real-time sleep quality predictions.
- **Additional Modalities:** Integrate other biosignals (e.g., EOG, EMG) to improve the model’s performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Happy coding! If you have any questions or suggestions, please open an issue or submit a pull request.*
```

---

Feel free to customize the sections and details according to your project's needs and the actual structure of your repository.

