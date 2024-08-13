# Depression Detection System

## Welcome!
The Depression Detection System is an open-source project developed on the MetaBCI platform, which is forked from the MetaBCI repository. This project combines EEG signals and deep learning techniques to effectively identify and analyze depressive states. The system is expected to play a significant role in clinical diagnosis assistance, mental health monitoring, and personalized treatment planning.

## Project Overview
Depression is a common and serious mental illness that significantly affects individuals' daily lives. Early identification and intervention are crucial for recovery. This project aims to develop a depression detection system using the MetaBCI platform, leveraging EEG signals and deep learning to recognize depressive patterns in brain activity.

## Content

- [Depression Detection System](#depression-detection-system)
  - [Welcome!](#welcome)
  - [Project Overview](#project-overview)
  - [What are we doing?](#what-are-we-doing)
    - [The problem](#the-problem)
    - [The solution](#the-solution)
  - [Features](#features)
  - [Installation](#installation)
  - [Who are we?](#who-are-we)
  - [What do we need?](#what-do-we-need)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## What are we doing?

### The problem

- Depression significantly impacts individuals' lives, and early identification is crucial for recovery.
- Existing methods for diagnosing depression are often subjective and lack objective biomarkers.
- There is a need for a system that combines EEG signals and deep learning to provide accurate and reliable depression detection.

### The solution

The Depression Detection System will:

- Utilize EEG signals to capture brain activity related to depression.
- Apply advanced preprocessing techniques to clean and normalize the EEG data.
- Extract meaningful features from the EEG signals.
- Train a 1D-CNN-GRU-ATTN model to classify depressive states with high accuracy.
- Provide visual feedback and detailed explanations of the results.

The goal of the Depression Detection System is to offer a reliable and efficient tool for early identification of depressive states, aiding in clinical diagnosis and treatment planning.

## Features

- **EEG Data Acquisition and Preprocessing**:
  - Real-time EEG signal acquisition using advanced EEG devices.
  - Comprehensive preprocessing steps including filtering, re-referencing, and artifact removal.

- **Feature Extraction**:
  - Power spectral density (PSD) and spectral features extraction.
  - Calculation of mean and standard deviation for each channel.

- **Model Training and Evaluation**:
  - Implementation of a 1D-CNN-GRU-ATTN model.
  - High accuracy, precision, recall, and F1 score in depressive state classification.

- **Visualization and Feedback**:
  - Generation of intuitive visual representations of depressive states.
  - Detailed text explanations and actionable feedback.


## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/xmanwo/DepressionDetection.MetaBCI.git
   ```
2. Change to the project directory
   ```sh
   cd DepressionDetection.MetaBCI
   ```
3. Install all requirements
   ```sh
   pip install -r requirements.txt 
   ```
4. Install brainda package with the editable mode
   ```sh
   pip install -e .
   ```


## Demos and Testing

The `demos` folder contains a key testing script, `addtest.py`, which is designed to validate and demonstrate the functionality of newly added features and optimizations to the MetaBCI platform. This includes:

- **Advanced Signal Processing**: Tests for adaptive filtering, wavelet decomposition, ICA reconstruction, sparse filtering, and more.
- **Deep Learning Model**: Validation of the 1D-CNN-GRU-ATTN model specifically designed for depression detection using EEG data.
- **Visualization Functions**: Comprehensive visualizations for EEG data including power spectral density (PSD), time-frequency analysis, and confusion matrices.
- **Optimized Data Processing**: Demonstrates the performance enhancements made through the use of optimized data handling, such as the `EnhancedProcessWorker` class.

This script is essential for verifying that the modifications and enhancements integrated into the MetaBCI platform function correctly and efficiently. It serves as a quick reference for developers and researchers to understand and test the added capabilities.


## Who are we?

The DepressionDetection project is carried out by researchers from 
- AI Brain-Computer Interface Laboratory, Institute of Applied Psychology, Beijing University of Technology, China

## What do we need?

**You**! In whatever way you can help.

We need expertise in programming, user experience, software sustainability, documentation, technical writing, and project management.

We'd love your feedback along the way.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. **Any contributions you make are greatly appreciated**. Especially welcome to submit BCI algorithms.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the GNU General Public License v2.0 License. See `LICENSE` for more information.

## Contact

Email: Chenxx@emails.bjut.edu.cn

## Acknowledgements
- [MetaBCI](https://github.com/TBC-TJU/MetaBCI)
- [MNE](https://github.com/mne-tools/mne-python)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [pyRiemann](https://github.com/alexandrebarachant/pyRiemann)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
