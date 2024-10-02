---
title: "Advancing Solar Flare Prediction: A Data Mining Approach"
subtitle: "Leveraging Machine Learning to Classify Solar Active Regions"
date: "12-2023"
---

<p align="center">[Github](https://github.com/karthikyeluripati/Solar_Flare_Prediction)</p>

## Introduction

Solar flares, intense bursts of radiation from the Sun's surface, have significant impacts on Earth's technological systems and space weather. Predicting these events accurately is crucial for mitigating potential risks to satellite communications, power grids, and astronaut safety. In this blog post, we'll delve into a cutting-edge data mining project aimed at developing a predictive model for solar flares based on historical solar magnetic field parameters.

## Project Overview

The primary objective was to classify solar active regions into two categories:

1. Flaring regions (X- and M-class flares)
2. Non-flaring regions (C-, B-, and Q-class flares)

This binary classification approach allows for a focused prediction of the most impactful solar flare events.

## Dataset and Preprocessing

### Data Collection

We obtained our dataset from two primary sources:
1. Kaggle
2. The Data Mining Lab at Georgia State University (https://dmlab.cs.gsu.edu/solar/data/data-comp-2020/)

The dataset comprises time series data of solar magnetic field parameters, categorized into five classes of solar flares: X, M, C, B, and Q.

### Data Preprocessing

To ensure the dataset was primed for modeling, we conducted thorough preprocessing steps:

1. **Handling Missing Values**: We employed mean imputation to address any gaps in the data, ensuring continuity in our time series.
2. **Outlier Detection**: We implemented robust statistical methods to identify and manage potential outliers that could skew our model's performance.

Here's a snippet of our preprocessing code:

```python
import pandas as pd
import numpy as np

def preprocess_data(data_path):
    # Load the data
    df = pd.read_json(data_path, lines=True)
    
    # Extract values from nested JSON
    df['values'] = df['values'].apply(pd.DataFrame.from_dict)
    
    # Handle missing values
    for col in df['values'].columns:
        df['values'][col] = df['values'][col].fillna(df['values'][col].mean())
    
    # Convert to numpy array
    values_np = np.array([v.to_numpy() for v in df['values']])
    
    return values_np, df['label']

# Usage
X, y = preprocess_data('train_partition1_data.json')
```

### Feature Engineering

Feature engineering played a pivotal role in capturing the complex behavior of solar magnetic fields. We extracted several types of features:

1. **Statistical Measures**: Including mean, median, standard deviation, and higher-order moments.
2. **Spectral Features**: Employing Fourier transforms to capture frequency-domain information.
3. **Temporal Patterns**: Extracting time-based features to capture the evolution of magnetic field parameters.

Here's a snippet demonstrating our feature extraction process:

```python
from scipy.stats import skew, kurtosis
from scipy.fft import fft

def extract_features(X):
    features = []
    for sample in X:
        sample_features = []
        # Statistical features
        sample_features.extend([np.mean(sample), np.median(sample), np.std(sample),
                                skew(sample), kurtosis(sample)])
        
        # Spectral features
        fft_vals = np.abs(fft(sample))
        sample_features.extend([np.max(fft_vals), np.mean(fft_vals)])
        
        # Temporal features
        sample_features.extend([np.gradient(sample).mean(), 
                                np.diff(sample).mean()])
        
        features.append(sample_features)
    
    return np.array(features)

# Usage
X_features = extract_features(X)
```

### Data Resampling

To address the inherent class imbalance, we implemented a sophisticated resampling strategy:

```python
from sklearn.utils import resample

def resample_data(X, y, sample_size):
    classes = np.unique(y)
    X_resampled, y_resampled = [], []
    
    for cls in classes:
        X_cls = X[y == cls]
        if len(X_cls) < sample_size:
            X_cls_resampled = resample(X_cls, n_samples=sample_size, random_state=42)
        else:
            X_cls_resampled = resample(X_cls, n_samples=sample_size, replace=False, random_state=42)
        X_resampled.extend(X_cls_resampled)
        y_resampled.extend([cls] * sample_size)
    
    return np.array(X_resampled), np.array(y_resampled)

# Usage
X_resampled, y_resampled = resample_data(X_features, y, sample_size=1000)
```

## Model Implementation and Evaluation

We implemented and compared two traditional machine learning models:

1. Canonical Interval Forest (CIF)
2. Dictionary-based model (WEASEL+MUSE)

### Canonical Interval Forest (CIF)

```python
from sktime.classification.interval_based import CanonicalIntervalForestClassifier

# Initialize and train the CIF model
cif = CanonicalIntervalForestClassifier(n_estimators=100, random_state=42)
cif.fit(X_train, y_train)

# Make predictions
y_pred_cif = cif.predict(X_test)
```

### Dictionary-based Model (WEASEL+MUSE)

```python
from sktime.classification.dictionary_based import WEASEL_MUSE

# Initialize and train the WEASEL+MUSE model
weasel_muse = WEASEL_MUSE(random_state=42)
weasel_muse.fit(X_train, y_train)

# Make predictions
y_pred_wm = weasel_muse.predict(X_test)
```

## Results and Performance Analysis

We evaluated both models using a comprehensive set of metrics:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_tss(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tss = (tp / (tp + fn)) - (fp / (fp + tn))
    return tss

def evaluate_model(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted'),
        'TSS': calculate_tss(y_true, y_pred)
    }

# Evaluate models
cif_metrics = evaluate_model(y_test, y_pred_cif)
wm_metrics = evaluate_model(y_test, y_pred_wm)

print("CIF Metrics:", cif_metrics)
print("WEASEL+MUSE Metrics:", wm_metrics)
```

The Dictionary-based model (WEASEL+MUSE) demonstrated superior performance across all metrics:

```
CIF Metrics: {
    'Accuracy': 0.8444,
    'Precision': 0.8502,
    'Recall': 0.8444,
    'F1 Score': 0.8456,
    'TSS': 0.6852
}

WEASEL+MUSE Metrics: {
    'Accuracy': 0.9211,
    'Precision': 0.9223,
    'Recall': 0.9211,
    'F1 Score': 0.9215,
    'TSS': 0.8383
}
```

These results indicate that the Dictionary-based model was more adept at distinguishing between flaring and non-flaring regions, with a particularly impressive true skill statistic that accounts for both true positive and true negative rates.

## Visualization of Results

To better understand our models' performance, we created visualizations of the results:

```python
import matplotlib.pyplot as plt

def plot_performance_comparison(cif_metrics, wm_metrics):
    metrics = list(cif_metrics.keys())
    cif_values = list(cif_metrics.values())
    wm_values = list(wm_metrics.values())

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, cif_values, width, label='CIF')
    ax.bar([i + width for i in x], wm_values, width, label='WEASEL+MUSE')

    ax.set_ylabel('Scores')
    ax.set_title('Performance Comparison: CIF vs WEASEL+MUSE')
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.show()

plot_performance_comparison(cif_metrics, wm_metrics)
```
![Performance Comparison: CIF vs WEASEL+MUSE](/images/projectpost-2/compatision.png)

## Challenges and Solutions

The primary challenge we encountered was the fragmented and imbalanced nature of the initial dataset. Our solution involved consolidating data from three separate files and implementing resampling techniques, as demonstrated in the code snippets above.

## Future Work

While our current models show promising results, there's potential for further improvement. We propose exploring advanced deep learning architectures in future iterations:

1. Convolutional Neural Networks (CNNs)
2. Multivariate Long Short-Term Memory Fully Convolutional Networks (MLSTM-FCN)

Here's a conceptual snippet of how we might implement an MLSTM-FCN model:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Concatenate, Dense, LSTM

def create_mlstm_fcn(input_shape, num_classes):
    ip = Input(shape=input_shape)
    
    # CNN branch
    conv1 = Conv1D(128, 8, padding='same', activation='relu')(ip)
    conv2 = Conv1D(256, 5, padding='same', activation='relu')(conv1)
    conv3 = Conv1D(128, 3, padding='same', activation='relu')(conv2)
    gap = GlobalAveragePooling1D()(conv3)
    
    # LSTM branch
    lstm = LSTM(128)(ip)
    
    # Merge
    concatenated = Concatenate()([gap, lstm])
    
    # Output
    out = Dense(num_classes, activation='softmax')(concatenated)
    
    model = Model(ip, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Usage
model = create_mlstm_fcn((timesteps, features), num_classes)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
```

## Conclusion

This project demonstrates the potential of data mining and machine learning techniques in advancing solar flare prediction. By leveraging sophisticated preprocessing, feature engineering, and modeling approaches, we've developed a system capable of distinguishing between flaring and non-flaring solar active regions with high accuracy.

The superior performance of the Dictionary-based model suggests that recurring patterns in solar magnetic field data play a crucial role in flare prediction. As we continue to refine these models and explore more advanced techniques, we move closer to a more reliable early warning system for potentially hazardous solar events.

This work not only contributes to the field of heliophysics but also has practical implications for safeguarding our increasingly technology-dependent society from the impacts of space weather.
