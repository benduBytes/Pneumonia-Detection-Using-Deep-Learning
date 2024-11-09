# Pneumonia Detection Using Deep Learning

## 🔧 Technologies & Tools
[![Tools](https://skillicons.dev/icons?i=python,anaconda,tensorflow,&perline=20)](https://skillicons.dev)

## Overview
This project leverages a convolutional neural network (CNN) for feature extraction from chest X-ray images, followed by training and prediction using various machine learning models. The system aims to provide a reliable tool for identifying pneumonia, potentially reducing diagnostic errors and supporting healthcare professionals.

## Features
- **Pneumonia Detection**: Uses a CNN for feature extraction from X-rays, combined with machine learning models for classification.
- **Feature Extraction with CNN**: Extracts relevant features from X-ray images, optimizing the data for machine learning classifiers.
- **Diverse ML Models for Classification**: Integrates multiple machine learning models, enhancing flexibility in prediction accuracy.
- **Threshold-Based Classification**: Classifies based on a probability threshold, enabling clear distinction between pneumonia-positive and negative cases.

## Dataset
- **Source**: Chest X-Ray Images (Pneumonia) dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data).
- **Content**: The dataset includes labeled X-ray images for pneumonia-positive and negative cases.

## Software and Libraries
- **Python**: Core programming language for model development.
- **Anaconda**: For package and environment management.
- **Jupyter Notebook**: Used as the primary development environment.

### Libraries
- **NumPy**: For handling arrays and numerical operations.
- **Pandas**: Data manipulation and analysis.
- **TensorFlow**: Core deep learning library for building the CNN model.
- **Scikit-Learn**: For implementing various machine learning models used in the final classification.
- **PIL**: Image processing.
- **Matplotlib**: Data visualization, for plotting model performance metrics.

## Model Architecture
Initially, the model used a CNN for both feature extraction and prediction, implemented in [Pneumonia_Detection_Using_CNN.ipynb](https://github.com/benduBytes/Pneumonia-Detection-Using-Deep-Learning/blob/main/Pneumonia_Detection_Using_CNN.ipynb). However, the current approach has shifted to use the CNN solely for feature extraction, with training and prediction handled by different machine learning models, as seen in the updated [Pneumonia_Detection_CNN_Features_ML_Model.ipynb](https://github.com/benduBytes/Pneumonia-Detection-Using-Deep-Learning/blob/main/Pneumonia_Detection_CNN_Features_ML_Model.ipynb).

## Preprocessing and Training
- **Preprocessing**: Images are resized to 256x256 pixels, and pixel values are normalized.
- **Feature Extraction**: Features are extracted from X-ray images using the CNN model.
- **Training**: Various machine learning models are trained on the extracted features for pneumonia classification.
- **Model Saving**: Trained models are saved for future use in real-time predictions.

## Expected Outcome
The model classifies X-ray images as:
- **Pneumonia-Positive**: Probability > 0.5
- **Pneumonia-Negative**: Probability ≤ 0.5

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/benduBytes/Pneumonia-Detection-Using-Deep-Learning
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project
1. **Prepare Dataset**: Ensure the dataset is downloaded from Kaggle and available in the specified directory.
2. **Run the Model**:
    ```bash
    python src/pneumonia_detection.py
    ```
   The system will load the X-ray images, process them, and classify each as pneumonia-positive or negative.

## Directory Structure
- **`Pneumonia Detection Using CNN.pdf`**: Project report and documentation.
- **`presentation/`**: Contains presentation slides related to the project.
- **`Pneumonia_Detection_CNN_Features_ML_Model.ipynb`**: Updated notebook implementing feature extraction using CNN and classification using machine learning models.
- **`Pneumonia_Detection_Using_CNN.ipynb`**: Original notebook implementing CNN for both feature extraction and prediction.
- **`README.md`**: Project README file.
- **`normal.jpg`**: Sample chest X-ray image (normal).
- **`pneumonic.jpg`**: Sample chest X-ray image (pneumonia).

## Issues and Future Improvements
- Model accuracy may vary based on image quality and dataset diversity.
- Potential improvements include adding more advanced architectures and increasing the dataset size for better generalization.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for enhancements, bug fixes, or new features.
